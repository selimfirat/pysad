FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:0aa4f12ee4e9a4aaa1e511b3ded82d662475150b82a786478f22d5c41adbacbd
COPY . $SRC/pysad
WORKDIR $SRC/pysad
COPY .clusterfuzzlite/build.sh $SRC/
