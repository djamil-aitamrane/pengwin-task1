FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    nnUNet_raw=/tmp/work/raw \
    nnUNet_preprocessed=/tmp/work/pp \
    nnUNet_results=/opt/ml/model

# compilateur pour les deps qui buildent (scipy/nnunet)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m -r -g user user
RUN mkdir -p /opt/app /input /output /tmp/work && chown -R user:user /opt/app /input /output /tmp/work

WORKDIR /opt/app
COPY --chown=user:user requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user inference.py /opt/app/inference.py
COPY --chown=user:user frac_to_instance.py /opt/app/frac_to_instance.py

USER user
ENTRYPOINT ["python", "inference.py"]

