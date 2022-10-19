FROM pytorchlightning/pytorch_lightning

RUN pip install numpy pandas matplotlib 

CMD /bin/bash