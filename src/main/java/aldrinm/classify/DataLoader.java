package aldrinm.classify;

import java.nio.file.Path;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * works off .npy files. tested  with > wget https://s3.amazonaws.com/img-datasets/mnist.npz -O
 * code/dlgo/nn/mnist.npz use a common zip utility to unzip the files in npz. they should all be
 * .npy files
 */
public final class DataLoader {

  private static final int NUM_CLASSES = 10;

  public static DataSet load(Path imagesPath, Path labelsPath) {
    INDArray images = Nd4j.createFromNpyFile(imagesPath.toFile());
    INDArray labels = Nd4j.createFromNpyFile(labelsPath.toFile());
    INDArray encodedLabels = Nd4j.create(DataType.FLOAT, labels.size(0), NUM_CLASSES);
    for (int i = 0; i < labels.size(0); i++) {
      encodedLabels.put(i, labels.getInt(i), 1f);
    }
    return new DataSet(images.castTo(DataType.FLOAT), encodedLabels);
  }

}
