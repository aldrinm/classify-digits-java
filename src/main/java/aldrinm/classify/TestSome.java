package aldrinm.classify;

import java.awt.image.BufferedImage;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TestSome {

  static void main() throws Exception {
    var data = Nd4j.arange(10);
    System.out.println("data = " + data);
    INDArray derivative = Transforms.sigmoidDerivative(data);
    System.out.println("derivative = " + derivative);

//    var f2 = first.reshape(3, 1, 2);
//    System.out.println("f2 = " + f2);
//
//// Dimensions: [BatchSize, RowsA, ColsA]
//    INDArray batchMatrixA = Nd4j.rand(new long[]{3, 3, 2});
//    System.out.println("batchMatrixA = " + batchMatrixA);
//// Dimensions: [ColsA, ColsB] (Rows of B must equal Cols of A)
//    INDArray matrixB = Nd4j.rand(new long[]{2, 1});
//    System.out.println("matrixB = " + matrixB);
//// Multiply across the 2nd axis of A (index 2) and 0th axis of B (index 0)
//    INDArray result = Nd4j.tensorMmul(batchMatrixA, matrixB, new int[][]{{2}, {0}});
//
//// The output shape will automatically be [BatchSize, RowsA, ColsB] -> [32, 4, 2]
//    System.out.println("result shape = "+ result.shapeInfo());
//    System.out.println("result = " + result);


    /* shuffling
    var first = Nd4j.rand(10, 2);
    var second = Nd4j.rand(10, 3);
    var dataset = new DataSet(first, second);
    dataset.shuffle();

    System.out.println("first = " + first);
    System.out.println("dataset.getFeatures() = " + dataset.getFeatures());
    System.out.println("second = " + second);
    System.out.println("dataset.getFeatures() = " + dataset.getLabels());
*/

/* simple loading
    Path dataDir = Path.of("./python");
    var trainingData = DataLoader.load(dataDir.resolve("x_test.npy"), dataDir.resolve("y_test.npy"));
*/


/*  //Dot product
    var a = Nd4j.create(new float[]{1, 2, 3});
//    var b = Nd4j.create(new float[][]{{4}, {5}, {6}});
    var b = Nd4j.create(new float[]{4, 5, 6});
    var r = Transforms.dot(a, b);
    System.out.println("r = " + r);
*/

//    DataSet testData = loadTestDataAndShowSomeStats();
//    DataSet trainingData = loadTrainDataAndShowSomeStats();

//    var averageEight = averageDigit(trainingData, 8);
    //    if (labels.getRow(i).argMax(0).getInt(0) == digit) {
//      return images.get(NDArrayIndex.point(i), NDArrayIndex.all(),
//          NDArrayIndex.all(),
//          NDArrayIndex.all());


//    averageDigit(testData, 7);
//    computeCloseness(trainingData);
  }

  private static void computeCloseness(DataSet dataSet) {
    //find image that matches 8
    INDArray labels = dataSet.getLabels();
    INDArray images = dataSet.getFeatures();

    var digit2 = images.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all(),
        NDArrayIndex.all());
    int anInt = labels.getRow(2).argMax(0).getInt(0);
    System.out.println("anInt = " + anInt);
    imgShow(digit2);
    var digit17 = images.get(NDArrayIndex.point(17), NDArrayIndex.all(), NDArrayIndex.all(),
        NDArrayIndex.all()); //digit is 8
    int anInt2 = labels.getRow(17).argMax(0).getInt(0);
    System.out.println("anInt2 = " + anInt2);
    imgShow(digit17);

    INDArray averageEight = averageDigit(dataSet, 8);
    var W = averageEight.transpose();

    Number dotTwo = W.mul(digit2).sumNumber();
    System.out.println("dot2 = " + dotTwo);
    Number dotSeventeen = W.mul(digit17).sumNumber();
    System.out.println("dot17 = " + dotSeventeen);

  }

  private static INDArray findDigitData(INDArray labels, INDArray images, int digit) {
    for (int i = 0; i< labels.size(0); i++) {
      if (labels.getRow(i).argMax(0).getInt(0) == digit) {
        return images.get(NDArrayIndex.point(i), NDArrayIndex.all(),
            NDArrayIndex.all(),
            NDArrayIndex.all());
      }
    }
    return null;
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
        bufferedImage.setRGB(x, y, (int) value);
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


  static INDArray averageDigit(DataSet testData, int digit) {
    System.out.println("features shape: " + testData.getFeatures().shapeInfoToString());
    System.out.println("label shape: " + testData.getLabels().shapeInfoToString());

    // images: [numberOfImages, 28, 28]
    var images = testData.getFeatures();
    var labels = testData.getLabels();
    // labels: [numberOfImages, 10], one-hot encoded
    INDArray sum = Nd4j.zeros( 28, 28);
    int count = 0;

    System.out.println("images.size(0) = " + images.size(0));
    for (int i = 0; i < images.size(0); i++) {
      if (labels.getRow(i).argMax(0).getInt(0) == digit) {
        sum.addi(images.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()));
        count++;
      }
    }

    if (count == 0) {
      throw new IllegalArgumentException("No images found for digit " + digit);
    }
    sum.divi(count);
    // Uncomment to see an image of the representation
    // imgShow(sum);
    return sum;
  }

}


