FROM continuumio/miniconda3

WORKDIR /work

# Create the environment:
COPY super-res.yml .
RUN conda env create -f super-res.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "super-res", "/bin/bash", "-c"]

# The code to run when container is started:
COPY ["srresnet.py", "jobs/start_job_nni_sing.sh", "./"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "super-res", "python", "srresnet.py"]
