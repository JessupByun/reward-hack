FROM python:3.12.16-slim-trixie@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && { python -VV; pip freeze; } > /etc/sandbox-versions.txt

# The check runner drops to this user to run pytest.
RUN useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin app

COPY harness/ /opt/harness/
RUN chmod -R u=rwX,go=rX /opt/harness && mkdir -p /repo/src /repo/tests

WORKDIR /repo
