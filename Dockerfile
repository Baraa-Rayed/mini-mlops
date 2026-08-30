# The task image — stage 2 of the local -> Docker -> Kubernetes journey.
#
# Note what this image does NOT contain: the scheduler, the CLI, the state
# database. It holds only what one *task* needs — an interpreter, the
# libraries, and the runner. That asymmetry is the whole design. The
# scheduler decides; the image executes; neither knows the other's internals.
#
#   docker build -t mini-mlops:latest .
#   python -m mini --executor docker run iris
#
# The code itself is bind-mounted at run time (see DockerExecutor), so editing
# a pipeline does not mean rebuilding this image. A production setup would
# COPY the code in instead, so that an image digest pins the code exactly —
# the trade is "fast iteration" against "reproducible by construction".

FROM python:3.11-slim

# Fail fast and log immediately: without this, a task's prints sit in a
# buffer and are lost if the container is killed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# No ENTRYPOINT on purpose. The executor supplies the full command line
# (`python -m mini.runner --ref ... --context ... --out ...`), which keeps the
# image a dumb, reusable box rather than something that only runs one task.
CMD ["python", "-c", "print('mini-mlops task image; the executor supplies the command')"]
