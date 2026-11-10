package aldrinm.classify;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActivationLayer extends Layer {

  private final int inputDim;
  private final int outputDim;

  public ActivationLayer(int inputDim) {
    super();
    this.inputDim = inputDim;
    this.outputDim = inputDim;
  }

  @Override
  void updateParams(double rate) {

  }

  @Override
  void forward() {
    INDArray data = this.getForwardInput();
    this.setOutputData(Transforms.sigmoid(data));
  }

  void backward() {
    var delta = this.getBackwardInput();
    var data = this.getForwardInput();
    this.setOutputDelta(delta.mul(Transforms.sigmoidDerivative(data)));
  }

  @Override
  void clearDeltas() {

  }

  @Override
  void describe() {
    System.out.printf("|-- %s%n", getClass().getSimpleName());
    System.out.printf("  |-- dimensions: (%s,%s)%n", inputDim, outputDim);
  }

  @Override
  public String toString() {
    return "ActivationLayer{" +
        "inputDim=" + inputDim +
        ", outputDim=" + outputDim +
        '}';
  }
}
