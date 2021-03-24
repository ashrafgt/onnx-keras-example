# ONNX Runtime Example: Simple Neural Network with Keras/Tensorflow

This example is a run-through of how to use the ONNX Runtime. Let's start by moving into the `keras-tf-nn` directory and following the steps.

### 1. Build the Docker image and run the container:
To run the demonstation build the image, and start the container:
```bash
docker build -t onnx-keras-nn -f Dockerfile.CUDA .
docker run -it --gpus=all onnx-keras-nn:latest bash
```
*If your device is not CUDA-capable, replace `Dockerfile.CUDA` with `Dockerfile.CPU` in the `build` command and remove `--gpus=all` from the `run` command.*

*If your device is CUDA-capable, make sure `nvidia-container-runtime` is installed.*


### 2. Train the model and test the default prediction:
As a start, train the Keras/Tensorflow model and save it:
```bash
export SAVED_MODEL_PATH="/mnt/models/simple_nn_saved_model"
python training/train.py $SAVED_MODEL_PATH  # model saved to the path we specified in the variable
python prediction/predict_default.py $SAVED_MODEL_PATH # note the prediction time output to the console
```

### 3. Convert the model to an ONNX graph:
To convert the saved Keras/Tensorflow model we use `tf2onnx` (`keras2onnx` is an alternative):
```bash
export CONVERTED_MODEL_PATH="/mnt/models/simple_nn_converted_model/converted_model.onnx"
python -m tf2onnx.convert \
    --saved-model $SAVED_MODEL_PATH \
    --output $CONVERTED_MODEL_PATH \
    --opset 12 # onnxruntime 1.7.0 supports up to ONNX OpSet 12
```

### 4. Run the ONNX prediction session:
Now that the model has been converted to an ONNX graph, and the default optimizations have been applied, we can test the prediction and review the performance boost:
```bash
python prediction/predict_onnx.py $CONVERTED_MODEL_PATH # note the very minor differences in the predicted values 
```