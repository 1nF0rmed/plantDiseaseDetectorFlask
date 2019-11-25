# A simple REST API for detecting diseases in Plants

## Model: Google Inception 2.0 (Pre-trained model fine tuned)

## Method: 

- Read camera feed
- Extract frame and store temporarily

![Step 2](step2.png)
- Send frame to REST API server

![Step 3](step3.png)
- Server processed image and inputs to Tensorflow Model

![Step 4](step4.png)
- Tensorflow Model provides predicted classification

![Step 5](step5.png)
- The plant state is then set to the classification

### Screenshot

![Output](output.png)
