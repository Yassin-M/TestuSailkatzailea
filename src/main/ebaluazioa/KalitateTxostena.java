package main.ebaluazioa;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Testu-sailkatzailearen kalitate-txostena sortzen duen utilitate klasea.
 * <p>
 * Klase honek bektorizatutako entrenamendu eta proba datuak kargatzen ditu,
 * BayesNet eredua konfigurazio onenarekin berreraiki (lehenik sailkatzaile hutsa kargatzen du)
 * eta entrenatzen du, eta sortzen duen ebaluazio eskema (kalitateTxostena.txt) fitxategian gorde.
 * </p>
 */
public class KalitateTxostena {
    /**
     * Kalitate txostenaren sorrera exekutatzen du.
     * <p>
     * Exekuzioan urrats hauek egiten dira:
     * entrenamendu/proba datuak kargatu, klase-indizea ezarri,
     * BayesNet eredu hutsa kargatu, konfigurazio-fitxategia irakurri, BayesNet eredua entrenatu,
     * ebaluazioa egin eta azken txostena fitxategian idatzi.
     * </p>
     *
     * @throws Exception Datu-fitxategiak, eredua edo konfigurazioa irakurri/erabiltzean
     *                   edo txostena idaztean gertatzen den edozein errore.
     */
    public static void kalitateaEstimatu(String trainBek, String testBek) throws Exception {
        //Test-aren importazioa
        DataSource sourceTestBektorizatua = new DataSource(testBek);
        Instances testBektorizatua = sourceTestBektorizatua.getDataSet();
        DataSource sourceTrainBektorizatua = new DataSource(trainBek);
        Instances trainBektorizatua = sourceTrainBektorizatua.getDataSet();

        if (testBektorizatua.classIndex() == -1) {
            testBektorizatua.setClassIndex(testBektorizatua.numAttributes() -1);
        }

        if (trainBektorizatua.classIndex() == -1) {
            trainBektorizatua.setClassIndex(trainBektorizatua.numAttributes() -1);
        }

        //Konfigurazio hoberena hartu
        String configuracionTxt = new String(Files.readAllBytes(Paths.get("/dataFinala/txt/config_bayes.txt")));

        //Eredu hutsik eta haren konfigurazio hoberena pasatu gero entrenatzeko
        BayesNet sailkatzailea = (BayesNet) SerializationHelper.read("/dataFinala/model/bestBayseNet.model");
        sailkatzailea.setOptions(Utils.splitOptions(configuracionTxt));
        sailkatzailea.buildClassifier(trainBektorizatua);

        //Ebaluazioa egin
        Evaluation eval = new Evaluation(trainBektorizatua);
        eval.evaluateModel(sailkatzailea, testBektorizatua);

        FileWriter fw = new FileWriter(new File("/dataFinala/txt/kalitateTxostena.txt"));

        //Ebaluzaio txostena izango duen parametroak hautatu (dokumentazioan azalpen konpletoago bat)
        fw.write("Kalitate txostena:");
        fw.write(eval.toMatrixString());

        //Klase balio bakoitzeko datuak
        fw.write("Klase bakoitzekiko ebaluazio datuak");
        //PREZISIOA
        for (int i = 0; i < testBektorizatua.classAttribute().numValues(); i++) {
            fw.write(testBektorizatua.classAttribute().value(i) + ": " + eval.precision(i));
        }
        fw.write("\n");

        //RECALL
        for (int i = 0; i < testBektorizatua.classAttribute().numValues(); i++) {
            fw.write(testBektorizatua.classAttribute().value(i) + ": " + eval.recall(i));
        }
        fw.write("\n");

        //F-SCORE
        for (int i = 0; i < testBektorizatua.classAttribute().numValues(); i++) {
            fw.write(testBektorizatua.classAttribute().value(i) + ": " + eval.fMeasure(i));
        }
        fw.write("\n");

        //Sailkatzaile globalen ebaluazioaren datuak (aurreko atalean aurkeztutako berdinak)
        fw.write("Sailkatzaile globalaren ebaluzaioaren datuak:\n");
        fw.write("Sailkatzailearen accuracy-a: " + eval.pctCorrect() + "%\n");
        fw.write("F-Score-ren batazbestekoa: " + eval.weightedFMeasure() + "\n");



        // Sailkatzailearen informazio gehigarria
        fw.write("\nSailkatzailearen informazioa:\n");
        fw.write("Erabilitako sailkatzaile mota: Bayes Network (BayesNet)\n");
        fw.write("Sailkatzailerako erabili diren parametro optimoak: " + weka.core.Utils.joinOptions(sailkatzailea.getOptions()) + "\n");
        String bektorizazioOptions = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("dataFinala/txt/bektorizazioHoberena.txt")));
        fw.write("Bektorizazioaren konfigurazioa: " + bektorizazioOptions + "\n");
        fw.write("Erabili den ebaluazio eskema: Hold-Out\n");

        fw.flush();
        fw.close();
    }
}