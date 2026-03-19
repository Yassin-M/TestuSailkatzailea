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

        //Ebaluzaio txostena izango duen parametroak hautatu


        //Kalitate txostena (ebaluzaioTxostena.txt) sortu aurrean aukeratu diren parametroekin
        //Behin txostena sortuta bezeroario emango diogun karpetan gorde

        FileWriter fw = new FileWriter(args[2]);
        //Txostena sortu
        fw.flush();
        fw.close();
    }
}
