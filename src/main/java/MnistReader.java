import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public final class MnistReader {

  private static final int IMAGE_MAGIC = 2051;
  private static final int LABEL_MAGIC = 2049;
  private static final int NUM_CLASSES = 10;

  private MnistReader() {
  }

  public static DataSet load(Path imagesPath, Path labelsPath)
      throws IOException {

    float[] images;
    int numImages;
    int rows;
    int cols;

    try (DataInputStream in = new DataInputStream(
        Files.newInputStream(imagesPath))) {

      int magic = in.readInt();
      numImages = in.readInt();
//      System.out.println("magic = " + magic);
//      System.out.println("numImages = " + numImages);
      rows = in.readInt();
      cols = in.readInt();
//      System.out.println("rows = " + rows+ " cols = " + cols );

      if (magic != IMAGE_MAGIC) {
        throw new IOException("Invalid MNIST image file magic: " + magic);
      }

      if (numImages <= 0 || rows <= 0 || cols <= 0) {
        throw new IOException("Invalid MNIST image dimensions");
      }

      int imageSize = Math.multiplyExact(rows, cols);
      int totalPixels = Math.multiplyExact(numImages, imageSize);
//      System.out.println("imageSize = " + imageSize);

      images = new float[totalPixels];

      for (int i = 0; i < totalPixels; i++) {
        int pixel = in.readUnsignedByte();
        images[i] = pixel / 255.0f;
      }

      if (in.read() != -1) {
        throw new IOException("Unexpected trailing bytes in image file");
      }
    }

    float[] labels = new float[Math.multiplyExact(numImages, NUM_CLASSES)];

    try (DataInputStream in = new DataInputStream(
        Files.newInputStream(labelsPath))) {

      int magic = in.readInt();
      int labelCount = in.readInt();

      if (magic != LABEL_MAGIC) {
        throw new IOException("Invalid MNIST label file magic: " + magic);
      }

      if (labelCount != numImages) {
        throw new IOException(
            "Image/label count mismatch: "
                + numImages + " images, "
                + labelCount + " labels");
      }

      for (int i = 0; i < labelCount; i++) {
        int label = in.readUnsignedByte();

        if (label >= NUM_CLASSES) {
          throw new IOException("Invalid digit label: " + label);
        }

        labels[i * NUM_CLASSES + label] = 1.0f;
      }

      if (in.read() != -1) {
        throw new IOException("Unexpected trailing bytes in label file");
      }
    } catch (EOFException e) {
      throw new IOException("Truncated MNIST label file", e);
    }

    // CNN input layout: [batch, channels, height, width].
    INDArray features = Nd4j.create(
        images,
        new long[]{numImages, 1, rows, cols}
    );

    // Classification targets: [batch, number of classes].
    INDArray targets = Nd4j.create(
        labels,
        new long[]{numImages, NUM_CLASSES}
    );

    return new DataSet(features, targets);
  }
}