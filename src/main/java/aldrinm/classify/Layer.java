package aldrinm.classify;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Layer {

  protected Layer previous;
  protected Layer next;
  private INDArray inputData;
  private INDArray outputData;
  private INDArray inputDelta;
  private INDArray outputDelta;

  public void setInputData(INDArray inputData) {
    this.inputData = inputData;
  }

  public INDArray getOutputData() {
    return outputData;
  }

  public void setOutputData(INDArray outputData) {
    this.outputData = outputData;
  }

  public void setInputDelta(INDArray inputDelta) {
    this.inputDelta = inputDelta;
  }

  public void setOutputDelta(INDArray outputDelta) {
    this.outputDelta = outputDelta;
  }

  void connect(Layer previousLayer) {
    previous = previousLayer;
    previousLayer.next = this;
  }

  abstract void forward();

  INDArray getForwardInput() {
    if (this.previous != null) {
      return this.previous.outputData;
    } else {
      return this.inputData;
    }
  }


  INDArray getBackwardInput() {
    if (this.next != null) {
      return this.next.outputDelta;
    } else {
      return this.inputDelta;
    }
  }

  abstract void backward();

  abstract void clearDeltas();

  abstract void updateParams(double rate);

  abstract void describe();

}
