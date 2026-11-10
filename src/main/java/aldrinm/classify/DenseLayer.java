package aldrinm.classify;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DenseLayer extends Layer {

  private final int inputDim;
  private final int outputDim;
  private final INDArray weight;
  private final List<INDArray> params;
  private INDArray bias;
  private INDArray deltaW;
  private INDArray deltaB;

  public DenseLayer(int inputDim, int outputDim) {
    super();
    this.inputDim = inputDim;
    this.outputDim = outputDim;

    this.weight = Nd4j.randn(outputDim, inputDim);
    this.bias = Nd4j.randn(outputDim);
    this.bias = this.bias.reshape(outputDim, 1);

    this.params = List.of(weight, bias);
    this.deltaW = Nd4j.zeros(weight.shape());
    this.deltaB = Nd4j.zeros(bias.shape());
    this.deltaB = this.deltaB.reshape(this.deltaB.size(0), 1);
  }

  @Override
  void forward() {
    var data = this.getForwardInput();
    var dot = weight.mmul(data);
    setOutputData(dot.add(bias));
  }

  @Override
  void backward() {
    var data = getForwardInput();
    var delta = getBackwardInput();
    deltaB.addi(delta);
    deltaW.addi(delta.mmul(data.transpose()));
    setOutputDelta(weight.transpose().mmul(delta));
  }

  @Override
  void updateParams(double rate) {
    weight.subi(deltaW.mul(rate));
    bias.subi(deltaB.mul(rate));
  }

  @Override
  void clearDeltas() {
    deltaW = Nd4j.zeros(weight.shape());
    deltaB = Nd4j.zeros(bias.shape());
  }

  @Override
  void describe() {
    System.out.printf("|--- %s%n", getClass().getSimpleName());
    System.out.printf(" |--- dimensions (%s,%s)%n", inputDim, outputDim);
  }

  @Override
  public String toString() {
    return "DenseLayer{" +
        "inputDim=" + inputDim +
        ", outputDim=" + outputDim +
        '}';
  }
}
