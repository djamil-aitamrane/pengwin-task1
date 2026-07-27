FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
RUN groupadd -r user && useradd -m -r -g user user
RUN mkdir -p /opt/app /input /output && chown -R user:user /opt/app /output
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir -r /opt/app/requirements.txt
COPY --chown=user:user inference.py frac_to_instance.py /opt/app/
USER user
WORKDIR /opt/app
ENTRYPOINT ["python", "inference.py"]
