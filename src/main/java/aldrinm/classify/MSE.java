package aldrinm.classify;

import org.nd4j.linalg.api.ndarray.INDArray;

public final class MSE {

  public static INDArray lossDerivative(INDArray predictions, INDArray labels) {
    return predictions.sub(labels);
  }
}
