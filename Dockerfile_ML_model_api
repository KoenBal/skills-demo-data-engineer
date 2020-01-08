FROM continuumio/miniconda3

MAINTAINER Koen Bal "koen.bal@gmail.com"

ADD environment_ML_model_api.yml /tmp/environment.yml
RUN conda env create --name custom-env --file /tmp/environment.yml \
    && conda clean -afy
     
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/envs/custom-env/bin:$PATH

# Copy all the files in your current working directory into the /app folder in the Docker container. 
# COPY . /app
COPY car_price_prediction_app.py /app/
COPY lgbr_cars.model /app/

WORKDIR /app

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "car_price_prediction_app.py" ]
