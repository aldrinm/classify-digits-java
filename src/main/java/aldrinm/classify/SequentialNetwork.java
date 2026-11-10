package aldrinm.classify;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MiniBatchFileDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SequentialNetwork {

  private final List<Layer> layers;
  private LossFunction loss; //not implemented

  public SequentialNetwork(LossFunction lossFunction) {
    System.out.println("Initializing Network...");
    layers = new LinkedList<>();
    this.loss = lossFunction;
    if (this.loss == null) {
      this.loss = LossFunction.MSE;
    }
  }

  public void add(Layer layer) {
    layers.add(layer);
    layer.describe();
    if (this.layers.size() > 1) {
      this.layers.getLast().connect(this.layers.get(layers.size() - 2));
    }
  }

  public void train(DataSet trainingData, int epochs, int miniBatchSize, double learningRate,
      DataSet testData) {
    trainingData.shuffle();
    for (int epoch = 0; epoch < epochs; epoch++) {
      System.out.println("Starting Epoch " + epoch);
      MiniBatchFileDataSetIterator dataSetIterator;
      try {
        dataSetIterator = new MiniBatchFileDataSetIterator(trainingData,
            miniBatchSize);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      var batchNum = 0;
      while (dataSetIterator.hasNext()) {
        batchNum++;
        if (batchNum % 100 == 0) {
          System.out.println(
              "Epoch " + epoch + " starting iter " + dataSetIterator.totalOutcomes()
                  + " batch number "
                  + batchNum);
        }
        DataSet next = dataSetIterator.next();
        var reshapedFeatures = next.getFeatures().reshape(next.getFeatures().size(0),
            next.getFeatures().size(1) * next.getFeatures().size(2));
        var reshapedLabels = next.getLabels()
            .reshape(next.getLabels().size(0), next.getLabels().size(1));
        DataSet reshaped = new DataSet(reshapedFeatures, reshapedLabels);
        trainBatch(reshaped, learningRate);
      }

      if (testData != null) {
        long evaluate = evaluate(testData);
        var nTest = testData.getFeatures().size(0);
        System.out.printf("Epoch %s: %s / %s %n", epoch, evaluate, nTest);
      } else {
        System.out.printf("Epoch %s complete", epoch);
      }
    }
  }

  private void trainBatch(DataSet batch, double learningRate) {
    forward_backward(batch);
    update(batch, learningRate);
  }

  private void update(DataSet batch, double learningRate) {
    var normalizedLearningRate = learningRate / batch.getFeatures().size(0);
    for (Layer layer : layers) {
      layer.updateParams(normalizedLearningRate);
    }
    for (Layer layer : layers) {
      layer.clearDeltas();
    }
  }

  private void forward_backward(DataSet batchSet) {
    //iterate through each set here
    for (DataSet dataSet : batchSet) {
      var features1 = dataSet.getFeatures()
          .reshape(dataSet.getFeatures().size(0) * dataSet.getFeatures().size(1), 1);
      var labels1 = dataSet.getLabels()
          .reshape(dataSet.getLabels().size(0) * dataSet.getLabels().size(1), 1);
      layers.getFirst().setInputData(features1);
      for (Layer layer : layers) {
        layer.forward();
      }

      INDArray lossDerivative = MSE.lossDerivative(layers.getLast().getOutputData(),
          labels1);
      layers.getLast().setInputDelta(lossDerivative);

      for (Layer layer : layers.reversed()) {
        layer.backward();
      }
    }
  }

  private INDArray singleForward(INDArray x) {
    layers.getFirst().setInputData(x);
    for (Layer layer : layers) {
      layer.forward();
    }
    return layers.getLast().getOutputData();
  }

  private long evaluate(DataSet testData) {
    long total = 0;
    for (DataSet testDatum : testData) {
      var features = testDatum.getFeatures()
          .reshape(testDatum.getFeatures().size(1) * testDatum.getFeatures().size(2), 1);

      INDArray predictedLabels = singleForward(features);
      INDArray indArray = predictedLabels.argMax();
      INDArray indArray1 = testDatum.getLabels().argMax();
      if (indArray.equals(indArray1)) {
        total++;
      }

    }
    return total;
  }

}
