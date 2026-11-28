FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:0aa4f12ee4e9a4aaa1e511b3ded82d662475150b82a786478f22d5c41adbacbd
COPY --from=ghcr.io/astral-sh/uv@sha256:3362a526af7eca2fcd8604e6a07e873fb6e4286d8837cb753503558ce1213664 /uv /usr/local/bin/uv
COPY . $SRC/pysad
WORKDIR $SRC/pysad
COPY .clusterfuzzlite/build.sh $SRC/
