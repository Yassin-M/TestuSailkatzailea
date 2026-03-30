package main.iragarpenak;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import main.datuak.CSV2Arff;
import main.datuak.Preprocessing;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Testu berrientzako iragarpenak egiten dituen exekuzio-klasea.
 * <p>
 * CSV sarrera batetik abiatuta, datuak ARFF formatura bihurtu eta garbitzen ditu (Aurreprozesamendua),
 * BayesNet eredua konfigurazio onenarekin entrenatzen du, eta iragarpenak
 * kontsolan zein irteera-fitxategi batean idazten ditu (iragarpenak.txt).
 * </p>
 */
public class Iragarpenak {

    /**
     * Iragarpen prozesu osoa exekutatzen du.
     * <p>
     * Lehenik, sarrerako CSV fitxategia aurreprozesatzen da; ondoren, entrenamendu
     * datu-sorta osatua prestatzen da, sailkatzailea berreraiki eta entrenatzen da,
     * eta azkenik test multzo blind-en iragarpenak sortu eta gordetzen dira.
     * </p>
     *
     * @param csvPath Exekuzio-argumentuak; testu gordinaren (.csv) path-a pasatzen da
     * @throws Exception Fitxategiak irakurtzean/idaztean, eredua kargatzean edo
     *                   Weka bidezko entrenamendu/ebaluazioan gertatutako errorea.
     */
    public static void iragarpenakEgin(String csvPath) throws Exception {
        System.out.println("Sarrerako datuak prestatzen...");

        //Hasi baino lehen, metodo honen input-etako bat testu gordina da (.csv). Beraz, aurreprozesamendu guztia
        //garatu behar da fitxategi horretan
        File f = new File(csvPath);
        String izenaArff = f.getName().replace(".csv", ".arff");
        String finalArffPath = "data/arff/raw/" + izenaArff;

        CSV2Arff.arffPasatu(csvPath);
        Preprocessing.tweetakGarbitu(finalArffPath);

        //Behin datu sorta garbi dagoela iragarpenak egiteko prest dago
        Instances testBlind = new DataSource(finalArffPath).getDataSet();
        if(testBlind.classIndex() == -1) testBlind.setClassIndex(testBlind.attribute("Sentiment").index());

        System.out.println("Datuak bektorizatzen eta atributuak filtratzen...");

        //Datu sorta osoa eta testBlind datuak bektorizatu
        String configTxt = new String(Files.readAllBytes(Paths.get("data/arff/bektorizatuta/txt/bektorizazioHoberena.txt")));
        StringToWordVector stwv = new StringToWordVector();
        stwv.setOptions(weka.core.Utils.splitOptions(configTxt));

        FixedDictionaryStringToWordVector fdstwv = new FixedDictionaryStringToWordVector();
        fdstwv.setTFTransform(stwv.getTFTransform());
        fdstwv.setIDFTransform(stwv.getIDFTransform());
        fdstwv.setOutputWordCounts(stwv.getOutputWordCounts());
        fdstwv.setTokenizer(stwv.getTokenizer());
        fdstwv.setStemmer(stwv.getStemmer());
        fdstwv.setDictionaryFile(new File("data/arff/bektorizatuta/txt/bestDictionary.txt"));

        fdstwv.setInputFormat(testBlind);
        Instances testBek = Filter.useFilter(testBlind, fdstwv);

        AttributeSelection as = (AttributeSelection) SerializationHelper.read("data/arff/bektorizatuta/filter/bestAttributeSelection.filter");
        Instances testFinal = Filter.useFilter(testBek, as);

        System.out.println("Eredua kargatzen eta iragarpenak egiten...");

        // Iragarpenak egin eredu finalarekin
        BayesNet eredua = (BayesNet) SerializationHelper.read("data/eredua/sailkatzaileFinala.model");

        Files.createDirectories(Paths.get("irteera"));
        try (FileWriter fw = new FileWriter("irteera/iragarpenak.txt")) {
            fw.write("Instance_ID,Predicted_Sentiment\n");

            for (int i = 0; i < testFinal.numInstances(); i++) {
                double prediccionIndex = eredua.classifyInstance(testFinal.instance(i));
                String prediccionLabel = testFinal.classAttribute().value((int) prediccionIndex);
                String testua = testBlind.instance(i).stringValue(testBlind.attribute("TweetText"));

                fw.write("ID:" + (i + 1) + " - " + prediccionLabel + ": " + testua + "\n");
            }
        }

        System.out.println("Iragarpenak arrakastaz gorde dira irteera/iragarpenak.txt fitxategian.");
    }
}
