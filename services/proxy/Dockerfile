FROM traefik:v2.11
RUN apk add --no-cache traefik gettext
COPY traefik /etc/traefik
ENTRYPOINT ["/bin/sh", "-c", \
    "envsubst < /etc/traefik/dynamic.yml > /etc/traefik/dynamic-subst.yml && \
    traefik --configFile /etc/traefik/traefik.yml" \
]
