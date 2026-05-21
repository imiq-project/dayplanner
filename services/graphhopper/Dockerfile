FROM eclipse-temurin:21-jre-alpine

ARG GRAPHHOPPER_VERSION=11.0

RUN apk add --no-cache curl bash

WORKDIR /graphhopper

RUN curl -fsSL \
    "https://repo1.maven.org/maven2/com/graphhopper/graphhopper-web/${GRAPHHOPPER_VERSION}/graphhopper-web-${GRAPHHOPPER_VERSION}.jar" \
    -o graphhopper.jar

RUN mkdir -p /cache
VOLUME ["/cache"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8989/health || exit 1

COPY ./cmd.sh ./config.yml ./
CMD [ "/graphhopper/cmd.sh" ]
