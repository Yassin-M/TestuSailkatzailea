package main;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;

public class KalitateTxostena {
    public static void main(String[] args) throws Exception {
        //Test-aren eta sailkatzailearen importazioa
        DataSource source = new DataSource(args[0]);
        Instances test = source.getDataSet();
        Classifier sailkatzailea = (Classifier) SerializationHelper.read(args[1]);
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() -1);
        }

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
        System.out.println("Sailkatzailerako erabili diren parametro optimoak: ");  //TODO CAMBIARLO PARA CUANDO (IBARRA) HAGA LO SUYO
        System.out.println("Bektorizazioaren konfigurazioa: "); //Esto igual lo quito

        //Kalitate txostena (ebaluzaioTxostena.txt) sortu aurrean aukeratu diren parametroekin
        //Behin txostena sortuta bezeroario emango diogun karpetan gorde

        FileWriter fw = new FileWriter(args[2]);
        //Txostena sortu
        fw.flush();
        fw.close();

        //TODO CAMBIAR TODOS LOS PRINTS A EL TEXTO QUE SE ESCRIBE
    }
}
