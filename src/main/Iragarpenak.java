package main;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import main.CSV2Arff;
import main.Preprocessing;

import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Iragarpenak {
    public static void main(String[] args) throws Exception {
        //Hasi baino lehen, metodo honen input-etako bat testu gordina da (.csv). Beraz, aurreprozesamendu guztia
        //garatu behar da fitxategi berean
        CSV2Arff.arffPasatu(args[0]);
        Preprocessing.tweetakGarbitu("./data/sortaGarbia.arff");

        //Behin datu sorta garbi dagoela iragarpenak egiteko prest dago
        DataSource source = new DataSource("data/clean/sortaGarbia.arff");
        Instances testBlind = source.getDataSet();

        if (testBlind.classIndex() == -1) {
            testBlind.setClassIndex(testBlind.numAttributes() - 1);
        }

        //Nahiz eta konfigurazio optimoa izan, sailkatzailea entrenatzeko datu sorta sortu behar da
        DataSource sourceTrain = new DataSource("./data/tweetSentiment.train.arff");
        Instances train = sourceTrain.getDataSet();
        DataSource sourceTest = new DataSource("./data/tweetSentiment.dev.arff");
        Instances test = sourceTrain.getDataSet();

        //Datu sortak unifikatu
        Instances datuOsoak = datuSortakUnifikatu(train, test);



        //Konfigurazio hoberena hartu
        String configuracionTxt = new String(Files.readAllBytes(Paths.get("config_bayes.txt")));

        //Eredu hutsik eta haren konfigurazio hoberena pasatu gero entrenatzeko
        BayesNet sailkatzailea = (BayesNet) SerializationHelper.read("./data/bestBayseNet.model");
        sailkatzailea.setOptions(Utils.splitOptions(configuracionTxt));


        //TODO FALTA LO DE VECTORIZAR

        sailkatzailea.buildClassifier(datuOsoak);

        //TODO FALTA HACER LO DE COMPROBAR E IGUALAR LOS HEADERS CON EL SAILKATZAILE

        //Emaitza horiek terminaletik inprimatu eta iragarpen fitxategi bat sortu emaitza hauek gordetzeko
        FileWriter fw = new FileWriter("./data/Iragarpenak.txt");

        if(test.equalHeaders(datuOsoak)) {
            //Ebaluazio aldagaia sortu eta main.sailkatzailea iragarri duen klaseak double-eko array batean gorde
            Evaluation eval = new Evaluation(testBlind);
            double[] iragarpenak = eval.evaluateModel(sailkatzailea, testBlind);

            for (int i = 0; i < iragarpenak.length; i++) {
                System.out.println("Iragarri den klasea: " + testBlind.attribute(testBlind.classIndex()).value((int) iragarpenak[i]));
                fw.write("Sailkatzailea hurrengo tweet-arako: " + testBlind.instance(i).stringValue(testBlind.numAttributes() -1) + " ----> Hurrengoa iragarri du: " + testBlind.attribute(testBlind.classIndex()).value((int) iragarpenak[i]));
                fw.write("\n");
            }
            fw.flush();
            fw.close();
        } else {
            fw.write("Header-ak ez dira berdina");
        }
    }

    public static Instances datuSortakUnifikatu (Instances train, Instances test) {
        int numDatuOsoak = train.numInstances() + test.numInstances();
        Instances datuOsoak =  new Instances(train, numDatuOsoak);

        for (int i = 0; i < train.numInstances(); i++) {
            datuOsoak.add(train.instance(i));
        }

        for (int i = 0; i < test.numInstances(); i++) {
            datuOsoak.add(test.instance(i));
        }
        return datuOsoak;
    }
}
