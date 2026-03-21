package main;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import main.CSV2Arff;

import java.io.FileWriter;

public class Iragarpenak {
    public static void main(String[] args) throws Exception {
        //Hasi baino lehen, metodo honen input-etako bat testu gordina da (.csv). Beraz, aurreprozesamendu guztia
        //pasatu beharko du .arff-an bihurtuz eta iragarpenak egin ahal izateko
        CSV2Arff.cleanCSV(args[0], args[1]);
        //TODO TODAVIA FALTA APLICAR EL OTRO PREPROCESAMIENTO DE LIMPIEZA DE DATOS (YASSIN)
        DataSource source = new DataSource(args[1]);
        Instances testBlind = source.getDataSet();

        if (testBlind.classIndex() == -1) {
            testBlind.setClassIndex(testBlind.numAttributes() - 1);
        }

        //Sailkatzailea kargatzen dugu
        Classifier sailkatzaile = (Classifier) SerializationHelper.read(args[2]);

        //TODO FALTA HACER LO DE COMPROBAR E IGUALAR LOS HEADERS CON EL SAILKATZAILE

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
