package aldrinm.classify;

import java.io.IOException;
import java.nio.file.Path;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class RunNetwork {

  static void main() throws IOException {
    System.out.println("Backend: " + Nd4j.getBackend().getClass().getSimpleName());
    // CPU:  CpuBackend
    // GPU:  JCublasBackend

    Path dataDir = Path.of("./python");

    DataSet trainingData = DataLoader.load(
        dataDir.resolve("x_train.npy"),
        dataDir.resolve("y_train.npy")
    );
    DataSet testData = DataLoader.load(
        dataDir.resolve("x_test.npy"),
        dataDir.resolve("y_test.npy")
    );

  /*
    //Subset of actual data for testing
    INDArray featureSlice = trainingData.getFeatures().get(NDArrayIndex.interval(0, 50),
        NDArrayIndex.all(), NDArrayIndex.all());
    INDArray labelSlice = trainingData.getLabels().get(NDArrayIndex.interval(0, 50));
    trainingData = new DataSet(featureSlice, labelSlice);

    var testFeatureSlice = testData.getFeatures().get(NDArrayIndex.interval(0, 50),
        NDArrayIndex.all(), NDArrayIndex.all());
    INDArray testLabelSlice = testData.getLabels().get(NDArrayIndex.interval(0, 50));
    testData = new DataSet(testFeatureSlice, testLabelSlice);
  */

    System.out.println("Loaded training data");
    SequentialNetwork network = new SequentialNetwork(null);

    network.add(new DenseLayer(784, 392));
    network.add(new ActivationLayer(392));
    network.add(new DenseLayer(392, 196));
    network.add(new ActivationLayer(196));
    network.add(new DenseLayer(196, 10));
    network.add(new ActivationLayer(10));

    network.train(trainingData, 10, 20, 3.0, testData);
    //network.train(trainingData, 10, 10, 3.0, null); //book

  }

}
