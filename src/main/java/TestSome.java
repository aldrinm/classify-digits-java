import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TestSome {

  static void main() throws IOException {

    DataSet testData = loadDataAndShowSomeStats();

    averageDigit(testData, 7);

  }

  private static DataSet loadDataAndShowSomeStats() throws IOException {
    Path dataDir = Path.of(".");

/* Training data takes long to load, so for trying things out just use the test data
    DataSet trainingData = MnistReader.load(
        dataDir.resolve("train-images.idx3-ubyte"),
        dataDir.resolve("train-labels.idx1-ubyte")
    );
    System.out.println("Training features: "
        + trainingData.getFeatures().shapeInfoToString());

    System.out.println("Training labels: "
        + trainingData.getLabels().shapeInfoToString());
*/

    DataSet testData = MnistReader.load(
        dataDir.resolve("t10k-images.idx3-ubyte"),
        dataDir.resolve("t10k-labels.idx1-ubyte")
    );

    System.out.println("Test features: "
        + testData.getFeatures().shapeInfoToString());

    INDArray singleImage = testData.getFeatures()
        .get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
    System.out.println("onerow.shapeInfoToString() = " + singleImage.shapeInfoToString());

//    imgShow(singleImage);

    return testData;
  }

  private static void imgShow(INDArray indArray) {

    INDArray image = indArray;
    while (image.rank() > 2) {
      image = image.get(NDArrayIndex.point(0));
    }

    int height = Math.toIntExact(image.rows());
    int width = Math.toIntExact(image.columns());

    BufferedImage bufferedImage = new BufferedImage(
        width, height, BufferedImage.TYPE_INT_RGB);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float value = image.getFloat(y, x);
        int gray = Math.round(Math.clamp(value, 0f, 1f) * 255f);
        bufferedImage.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
      }
    }

    //Display the image in a Java Swing JFrame
    JFrame frame = new JFrame();
    frame.setSize(bufferedImage.getWidth(), bufferedImage.getHeight());
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.getContentPane().add(new JLabel(new ImageIcon(bufferedImage)));
    frame.pack();
    frame.setVisible(true);
  }


  static void averageDigit(DataSet testData, int digit) {
    System.out.println("features shape: " + testData.getFeatures().shapeInfoToString());
    System.out.println("label shape: " + testData.getLabels().shapeInfoToString());

    // images: [numberOfImages, 784], with pixel values in [0, 1]
    var images = testData.getFeatures();
    var labels = testData.getLabels();
    // labels: [numberOfImages, 10], one-hot encoded
    INDArray sum = Nd4j.zeros(1, 28, 28);
    int count = 0;

    System.out.println("images.size(0) = " + images.size(0));
    for (int i = 0; i < images.size(0); i++) {
      if (labels.getRow(i).argMax(0).getInt(0) == digit) {
        sum.addi(images.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
            NDArrayIndex.all()));
        count++;
      }
    }

    if (count == 0) {
      throw new IllegalArgumentException("No images found for digit " + digit);
    }
    sum.divi(count);
    imgShow(sum);
  }

}


