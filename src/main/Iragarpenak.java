package main;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;

public class Iragarpenak {
    public static void main(String[] args) throws Exception {
        //Test blind DataSet-a eta iragarpenak egiteko erabiliko den sailkatzailea importatu
        DataSource source = new DataSource(args[0]);
        Instances testBlind = source.getDataSet();
        Classifier sailkatzaile = (Classifier) SerializationHelper.read(args[1]);

        //Ebaluazio aldagaia sortu eta sailkatzailea iragarri duen klaseak double-eko array batean gorde
        Evaluation eval = new Evaluation(testBlind);
        double[] iragarpenak = eval.evaluateModel(sailkatzaile, testBlind);

        //Emaitza horiek terminaletik inprimatu eta iragarpen fitxategi bat sortu emaitza hauek gordetzeko
        FileWriter fw = new FileWriter(args[2]);
        for (int i = 0; i < iragarpenak.length; i++) {
            System.out.println("Iragarri den klasea: " + testBlind.attribute(testBlind.classIndex()).value((int) iragarpenak[i]));
            fw.write("Iragarri den klasea: " + testBlind.attribute(testBlind.classIndex()).value((int) iragarpenak[i]));
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }
}
