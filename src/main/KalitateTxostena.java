package main;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

public class KalitateTxostena {
    public static void main(String[] args) throws Exception {
        //Test-aren importazioa
        DataSource source = new DataSource("./data/tweetSentiment.dev.arff");
        Instances test = source.getDataSet();

        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() -1);
        }

        //Konfigurazio hoberena hartu
        String configuracionTxt = new String(Files.readAllBytes(Paths.get("config_bayes.txt")));

        //Eredu hutsik eta haren konfigurazio hoberena pasatu gero entrenatzeko
        BayesNet sailkatzailea = (BayesNet) SerializationHelper.read("./data/bestBayseNet.model");
        sailkatzailea.setOptions(Utils.splitOptions(configuracionTxt));
        sailkatzailea.buildClassifier(test);

        //TODO BEKTORIZATU EREDUA

        //Ebaluazioa egin
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(sailkatzailea, test);

        //Ebaluzaio txostena izango duen parametroak hautatu (dokumentazioan azalpen konpletoago bat)
        System.out.println("Kalitate txostena:");
        System.out.println(eval.toMatrixString());

        //Klase balio bakoitzeko datuak
        System.out.println("Klase bakoitzekiko ebaluazio datuak");
        //PREZISIOA
        for (int i = 0; i < test.classAttribute().numValues(); i++) {
            System.out.println(test.classAttribute().value(i) + ": " + eval.precision(i));
        }
        System.out.println("\n");

        //RECALL
        for (int i = 0; i < test.classAttribute().numValues(); i++) {
            System.out.println(test.classAttribute().value(i) + ": " + eval.recall(i));
        }
        System.out.println("\n");

        //F-SCORE
        for (int i = 0; i < test.classAttribute().numValues(); i++) {
            System.out.println(test.classAttribute().value(i) + ": " + eval.fMeasure(i));
        }
        System.out.println("\n");

        //Sailkatzaile globalen ebaluazioaren datuak (aurreko atalean aurkeztutako berdinak)
        System.out.println("Sailkatzaile globalaren ebaluzaioaren datuak:");
        System.out.println("Sailkatzailearen accuracy-a: " + eval.pctCorrect());
        System.out.println("F-Score-ren batazbestekoa: " + eval.weightedFMeasure());

        //Sailkatzailearen informazio gehigarria
        System.out.println("Sailkatzailearen informazioa: ");
        System.out.println("Erabilitako sailkatzaile mota: Bayes Network (BayesNet)");
        System.out.println("Sailkatzailerako erabili diren parametro optimoak: " + sailkatzailea.getOptions());
        System.out.println("Bektorizazioaren konfigurazioa: "); //Esto igual lo quito
        System.out.println("Erabili den ebaluazio eskema: Hold-Out ");

        //Kalitate txostena (ebaluzaioTxostena.txt) sortu aurrean aukeratu diren parametroekin
        //Behin txostena sortuta bezeroario emango diogun karpetan gorde

        FileWriter fw = new FileWriter(args[2]);
        //Txostena sortu
        fw.flush();
        fw.close();

        //TODO CAMBIAR TODOS LOS PRINTS A EL TEXTO QUE SE ESCRIBE
    }
}
