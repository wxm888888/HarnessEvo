#define _GNU_SOURCE

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define HEADER_MAX 262144
#define IO_BUF 65536

static volatile sig_atomic_t stopping = 0;
static int sequence_no = 0;

static void stop_signal(int signo) { (void)signo; stopping = 1; }

static int write_all(int fd, const void *data, size_t length) {
    const char *p = (const char *)data;
    while (length > 0) {
        ssize_t n = send(fd, p, length, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += n;
        length -= (size_t)n;
    }
    return 0;
}

static int write_file(FILE *file, const void *data, size_t length) {
    return length == 0 || fwrite(data, 1, length, file) == length ? 0 : -1;
}

static ssize_t read_some(int fd, void *data, size_t length) {
    for (;;) {
        ssize_t n = recv(fd, data, length, 0);
        if (n < 0 && errno == EINTR) continue;
        return n;
    }
}

static int read_exact(int fd, void *data, size_t length) {
    char *p = (char *)data;
    while (length > 0) {
        ssize_t n = read_some(fd, p, length);
        if (n <= 0) return -1;
        p += n;
        length -= (size_t)n;
    }
    return 0;
}

static int read_headers(int fd, char *buffer, size_t capacity, size_t *out_len) {
    size_t used = 0;
    while (used + 1 < capacity) {
        ssize_t n = read_some(fd, buffer + used, 1);
        if (n <= 0) return -1;
        used += (size_t)n;
        if (used >= 4 && memcmp(buffer + used - 4, "\r\n\r\n", 4) == 0) {
            buffer[used] = '\0';
            *out_len = used;
            return 0;
        }
    }
    return -1;
}

static char *find_header_value(char *headers, const char *name) {
    size_t name_len = strlen(name);
    char *line = headers;
    while (line && *line) {
        char *end = strstr(line, "\r\n");
        size_t length = end ? (size_t)(end - line) : strlen(line);
        if (length > name_len && strncasecmp(line, name, name_len) == 0 && line[name_len] == ':') {
            char *value = line + name_len + 1;
            while (*value == ' ' || *value == '\t') value++;
            return value;
        }
        if (!end) break;
        line = end + 2;
    }
    return NULL;
}

static void session_header(char *headers, const char *name, char *dest, size_t cap) {
    dest[0] = '\0';
    const char *value = find_header_value(headers, name);
    if (!value) return;
    size_t length = strcspn(value, "\r\n");
    while (length && (value[length - 1] == ' ' || value[length - 1] == '\t')) length--;
    if (!length || length >= cap) return;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)value[i];
        if (!(isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':')) return;
    }
    memcpy(dest, value, length);
    dest[length] = '\0';
}

static long long header_length(char *headers) {
    char *value = find_header_value(headers, "Content-Length");
    if (!value) return -1;
    return strtoll(value, NULL, 10);
}

static int header_chunked(char *headers) {
    char *value = find_header_value(headers, "Transfer-Encoding");
    return value && strcasestr(value, "chunked") != NULL;
}

static int path_is_generation(const char *method, const char *path) {
    size_t n = strlen(path);
    const char *chat = "/chat/completions";
    const char *responses = "/responses";
    return strcasecmp(method, "POST") == 0 &&
           ((n >= strlen(chat) && strcmp(path + n - strlen(chat), chat) == 0) ||
            (n >= strlen(responses) && strcmp(path + n - strlen(responses), responses) == 0));
}

static int json_bool_true(const char *body, size_t length, const char *key) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = body;
    const char *end = body + length;
    while (p < end) {
        const char *hit = strstr(p, needle);
        if (!hit || hit >= end) return 0;
        const char *colon = strchr(hit + strlen(needle), ':');
        if (!colon || colon >= end) return 0;
        colon++;
        while (colon < end && isspace((unsigned char)*colon)) colon++;
        if (strncmp(colon, "true", 4) == 0) return 1;
        p = hit + strlen(needle);
    }
    return 0;
}

static void extract_request_line(const char *headers, char *method, size_t method_cap,
                                 char *path, size_t path_cap) {
    method[0] = path[0] = '\0';
    sscanf(headers, "%31s %1023s", method, path);
    method[method_cap - 1] = '\0';
    path[path_cap - 1] = '\0';
}

static int parse_url(const char *url, char *host, size_t host_cap, char *port,
                     size_t port_cap, char *base_path, size_t path_cap) {
    const char *start = strstr(url, "://");
    if (!start || strncasecmp(url, "http://", 7) != 0) return -1;
    start += 3;
    const char *slash = strchr(start, '/');
    const char *authority_end = slash ? slash : start + strlen(start);
    const char *colon = NULL;
    for (const char *p = start; p < authority_end; p++) if (*p == ':') colon = p;
    if (colon) {
        size_t host_len = (size_t)(colon - start);
        if (host_len == 0 || host_len >= host_cap) return -1;
        memcpy(host, start, host_len);
        host[host_len] = '\0';
        snprintf(port, port_cap, "%.*s", (int)(authority_end - colon - 1), colon + 1);
    } else {
        size_t host_len = (size_t)(authority_end - start);
        if (host_len == 0 || host_len >= host_cap) return -1;
        memcpy(host, start, host_len);
        host[host_len] = '\0';
        snprintf(port, port_cap, "80");
    }
    if (slash) snprintf(base_path, path_cap, "%s", slash);
    else snprintf(base_path, path_cap, "/");
    return 0;
}

static int connect_host(const char *host, const char *port) {
    struct addrinfo hints, *result = NULL, *item;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host, port, &hints, &result) != 0) return -1;
    int fd = -1;
    for (item = result; item; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, item->ai_addr, item->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    return fd;
}

static int append_path(char *out, size_t cap, const char *base, const char *path) {
    if (strcmp(base, "/") == 0) return snprintf(out, cap, "%s", path) < (int)cap ? 0 : -1;
    if (strncmp(path, base, strlen(base)) == 0 &&
        (path[strlen(base)] == '/' || path[strlen(base)] == '\0')) {
        return snprintf(out, cap, "%s", path) < (int)cap ? 0 : -1;
    }
    return snprintf(out, cap, "%s%s", base, path) < (int)cap ? 0 : -1;
}

static int forward_request(int upstream, const char *headers, size_t headers_len,
                           const char *body, size_t body_len, const char *host,
                           const char *port) {
    const char *first_end = strstr(headers, "\r\n");
    if (!first_end) return -1;
    if (write_all(upstream, headers, (size_t)(first_end - headers) + 2) < 0) return -1;
    const char *line = first_end + 2;
    while (line < headers + headers_len && !(line[0] == '\r' && line[1] == '\n')) {
        const char *end = strstr(line, "\r\n");
        if (!end) return -1;
        size_t length = (size_t)(end - line);
        int skip = (length >= 5 && strncasecmp(line, "Host:", 5) == 0) ||
                   (length >= 11 && strncasecmp(line, "Connection:", 11) == 0) ||
                   (length >= 17 && strncasecmp(line, "Proxy-Connection:", 17) == 0);
        if (!skip && write_all(upstream, line, length + 2) < 0) return -1;
        line = end + 2;
    }
    if (dprintf(upstream, "Host: %s%s%s\r\nConnection: close\r\n\r\n",
                host, strcmp(port, "80") == 0 ? "" : ":",
                strcmp(port, "80") == 0 ? "" : port) < 0) return -1;
    return body_len == 0 || write_all(upstream, body, body_len) == 0 ? 0 : -1;
}

static int read_line(int fd, char *line, size_t cap, size_t *out_len) {
    size_t used = 0;
    while (used + 1 < cap) {
        ssize_t n = read_some(fd, line + used, 1);
        if (n <= 0) return -1;
        used++;
        if (used >= 2 && line[used - 2] == '\r' && line[used - 1] == '\n') {
            line[used] = '\0';
            *out_len = used;
            return 0;
        }
    }
    return -1;
}

static int proxy_body(int upstream, int client, char *response_headers, size_t response_len,
                      FILE *capture) {
    if (write_all(client, response_headers, response_len) < 0) {
        /* Continue consuming the upstream response so the capture remains complete. */
    }
    int chunked = header_chunked(response_headers);
    long long length = header_length(response_headers);
    char buffer[IO_BUF];
    if (chunked) {
        char line[8192];
        for (;;) {
            size_t line_len = 0;
            if (read_line(upstream, line, sizeof(line), &line_len) < 0) return -1;
            (void)write_all(client, line, line_len);
            unsigned long long chunk = strtoull(line, NULL, 16);
            if (chunk == 0) {
                do {
                    if (read_line(upstream, line, sizeof(line), &line_len) < 0) return -1;
                    (void)write_all(client, line, line_len);
                } while (!(line_len == 2 && line[0] == '\r' && line[1] == '\n'));
                return 0;
            }
            unsigned long long remaining = chunk;
            while (remaining > 0) {
                size_t want = remaining > sizeof(buffer) ? sizeof(buffer) : (size_t)remaining;
                if (read_exact(upstream, buffer, want) < 0) return -1;
                (void)write_all(client, buffer, want);
                if (write_file(capture, buffer, want) < 0) return -1;
                remaining -= want;
            }
            char crlf[2];
            if (read_exact(upstream, crlf, 2) < 0) return -1;
            (void)write_all(client, crlf, 2);
        }
    }
    if (length >= 0) {
        long long remaining = length;
        while (remaining > 0) {
            size_t want = remaining > (long long)sizeof(buffer) ? sizeof(buffer) : (size_t)remaining;
            if (read_exact(upstream, buffer, want) < 0) return -1;
            (void)write_all(client, buffer, want);
            if (write_file(capture, buffer, want) < 0) return -1;
            remaining -= (long long)want;
        }
        return 0;
    }
    for (;;) {
        ssize_t n = read_some(upstream, buffer, sizeof(buffer));
        if (n == 0) return 0;
        if (n < 0) return -1;
        (void)write_all(client, buffer, (size_t)n);
        if (write_file(capture, buffer, (size_t)n) < 0) return -1;
    }
}

static void write_meta(const char *path, int n, const char *method, const char *request_path,
                       int status, int stream, const char *session_id, const char *parent_session_id) {
    FILE *file = fopen(path, "w");
    if (!file) return;
    fprintf(file, "{\"n\":%d,\"method\":\"%s\",\"path\":\"%s\",\"status\":%d,\"stream\":%s,\"session_id\":\"%s\",\"parent_session_id\":\"%s\"}\n",
            n, method, request_path, status, stream ? "true" : "false", session_id, parent_session_id);
    fclose(file);
}

static int handle_client(int client, const char *upstream_url, const char *outdir) {
    char request_headers[HEADER_MAX];
    size_t request_len = 0;
    if (read_headers(client, request_headers, sizeof(request_headers), &request_len) < 0) return -1;
    char method[32], request_path[2048];
    extract_request_line(request_headers, method, sizeof(method), request_path, sizeof(request_path));
    char session_id[256], parent_session_id[256];
    session_header(request_headers, "X-Session-Id", session_id, sizeof(session_id));
    if (!session_id[0]) session_header(request_headers, "x-opencode-session", session_id, sizeof(session_id));
    if (!session_id[0]) session_header(request_headers, "x-session-affinity", session_id, sizeof(session_id));
    session_header(request_headers, "x-parent-session-id", parent_session_id, sizeof(parent_session_id));
    long long content_length = header_length(request_headers);
    if (content_length < 0) content_length = 0;
    if (content_length > 64LL * 1024 * 1024) return -1;
    char *body = NULL;
    if (content_length > 0) {
        body = malloc((size_t)content_length);
        if (!body || read_exact(client, body, (size_t)content_length) < 0) {
            free(body);
            return -1;
        }
    }
    int capture = path_is_generation(method, request_path);
    int n = capture ? ++sequence_no : 0;
    int stream = capture && body && json_bool_true(body, (size_t)content_length, "stream");
    if (capture) mkdir(outdir, 0755);
    char host[512], port[32], base_path[2048], target_path[4096];
    if (parse_url(upstream_url, host, sizeof(host), port, sizeof(port), base_path, sizeof(base_path)) < 0 ||
        append_path(target_path, sizeof(target_path), base_path, request_path) < 0) {
        free(body);
        return -1;
    }
    char *request_copy = NULL;
    size_t request_copy_len = request_len;
    request_copy = malloc(request_len + 1);
    if (!request_copy) { free(body); return -1; }
    memcpy(request_copy, request_headers, request_len + 1);
    char *first_end = strstr(request_copy, "\r\n");
    if (!first_end) { free(body); free(request_copy); return -1; }
    char first_line[4096];
    snprintf(first_line, sizeof(first_line), "%s %s HTTP/1.1\r\n", method, target_path);
    size_t first_len = strlen(first_line);
    size_t tail_len = request_copy_len - (size_t)(first_end - request_copy) - 2;
    memmove(request_copy + first_len, first_end + 2, tail_len);
    memcpy(request_copy, first_line, first_len);
    request_copy_len = first_len + tail_len;
    int upstream = connect_host(host, port);
    char req_path[4096];
    req_path[0] = '\0';
    if (capture) {
        snprintf(req_path, sizeof(req_path), "%s/%03d-req.json", outdir, n);
        FILE *req_file = fopen(req_path, "wb");
        if (!req_file || write_file(req_file, body, (size_t)content_length) < 0) {
            if (req_file) fclose(req_file);
            free(body); free(request_copy);
            if (upstream >= 0) close(upstream);
            return -1;
        }
        fclose(req_file);
    }
    if (upstream < 0 || forward_request(upstream, request_copy, request_copy_len, body,
                                         (size_t)content_length, host, port) < 0) {
        if (upstream >= 0) close(upstream);
        free(body); free(request_copy);
        return -1;
    }
    free(body); free(request_copy);
    char response_headers[HEADER_MAX];
    size_t response_len = 0;
    if (read_headers(upstream, response_headers, sizeof(response_headers), &response_len) < 0) {
        close(upstream);
        return -1;
    }
    int status = 0;
    sscanf(response_headers, "HTTP/%*s %d", &status);
    FILE *capture_file = NULL;
    char tmp_path[4096], final_path[4096], meta_path[4096];
    if (capture) {
        mkdir(outdir, 0755);
        snprintf(tmp_path, sizeof(tmp_path), "%s/%03d-resp.tmp", outdir, n);
        snprintf(final_path, sizeof(final_path), "%s/%03d-resp.%s", outdir, n, stream ? "sse" : "json");
        snprintf(meta_path, sizeof(meta_path), "%s/%03d-meta.json", outdir, n);
    }
    int rc;
    if (capture) {
        capture_file = fopen(tmp_path, "wb");
        if (!capture_file) rc = -1;
        else rc = proxy_body(upstream, client, response_headers, response_len, capture_file);
    } else {
        rc = proxy_body(upstream, client, response_headers, response_len, tmpfile());
    }
    if (capture_file) {
        fflush(capture_file);
        fclose(capture_file);
        if (rc == 0) {
            rename(tmp_path, final_path);
            write_meta(meta_path, n, method, request_path, status, stream, session_id, parent_session_id);
        } else unlink(tmp_path);
    }
    close(upstream);
    return rc;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    /*
     * signal() may install handlers with SA_RESTART, which leaves accept()
     * blocked after SIGTERM until another client happens to connect.  The
     * rollout wrapper waits for this process during teardown, so that race
     * turns an already-complete sample into an agent timeout.  Use sigaction
     * without SA_RESTART so the stop signal reliably interrupts accept().
     */
    struct sigaction stop_action;
    memset(&stop_action, 0, sizeof(stop_action));
    stop_action.sa_handler = stop_signal;
    sigemptyset(&stop_action.sa_mask);
    if (sigaction(SIGTERM, &stop_action, NULL) < 0 ||
        sigaction(SIGINT, &stop_action, NULL) < 0) return 5;
    const char *upstream = getenv("TRACE_PROXY_UPSTREAM");
    const char *outdir = getenv("TRACE_PROXY_OUTDIR");
    const char *port_text = getenv("TRACE_PROXY_PORT");
    if (!upstream || !*upstream) {
        fprintf(stderr, "TRACE_PROXY_UPSTREAM is required\n");
        return 2;
    }
    if (!outdir || !*outdir) outdir = "/logs/agent/llm-reqs";
    /*
     * TRACE_PROXY_PORT=0 asks the kernel for an ephemeral loopback port.
     * The launcher uses this mode so a task that starts its own service on
     * an arbitrary port cannot collide with the trace proxy.
     */
    int port = port_text ? atoi(port_text) : 8765;
    if (port < 0 || port > 65535) {
        fprintf(stderr, "invalid TRACE_PROXY_PORT=%s\n", port_text ? port_text : "");
        return 2;
    }
    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return 3;
    int one = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t)port);
    if (bind(server, (struct sockaddr *)&address, sizeof(address)) < 0 || listen(server, 32) < 0) {
        close(server);
        return 4;
    }
    struct sockaddr_in bound_address;
    socklen_t bound_length = sizeof(bound_address);
    memset(&bound_address, 0, sizeof(bound_address));
    if (getsockname(server, (struct sockaddr *)&bound_address, &bound_length) < 0) {
        close(server);
        return 4;
    }
    int actual_port = (int)ntohs(bound_address.sin_port);
    mkdir(outdir, 0755);
    fprintf(stderr, "trace-proxy-ready upstream=%s outdir=%s port=%d\n", upstream, outdir, actual_port);
    fflush(stderr);
    while (!stopping) {
        int client = accept(server, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR && !stopping) continue;
            break;
        }
        handle_client(client, upstream, outdir);
        close(client);
    }
    close(server);
    return 0;
}
