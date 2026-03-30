package main.datuak;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

/**
 * Tweet-en datu-sortak sailkapenerako prestatzen dituen klasea.
 * ARFF fitxategiak kargatu, atributu ez-beharrezkoak ezabatu eta testuaren garbiketa sakona egiten du zarata gutxitzeko.
 */
public class Preprocessing {

    /**
     * ARFF fitxategi batetik datuak kargatu, informazio garrantzitsuarekin geratu eta Tweet-en testua banan-banan garbitzen du.
     * Hurrengo pausoak egiten ditu:
     * <ul>
     * <li>Fitxategia kargatu eta 'TweetId' eta 'TweetDate' atributuak ezabatu.</li>
     * <li>Instantzia bakoitzeko testua {@link #cleanTweet(String)} bidez garbitu.</li>
     * <li>Garbiketaren ostean hutsik geratzen diren instantziak datu-sortatik kendu.</li>
     * <li>Emaitza jatorrizko fitxategi bidean (path) gainidatzi.</li>
     * </ul>
     * @param path Fitxategiaren helbide erlatibo edo absolutua (path).
     * @throws Exception Fitxategia kargatzean, gordetzean edo iragaztean akatsen bat gertatuz gero.
     */
    public static void tweetakGarbitu(String path) throws Exception {
        int ezabatutakoakCount = 0;

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();

        Remove rm = new Remove();
        rm.setAttributeIndicesArray(new int[]{data.attribute("TweetId").index(), data.attribute("TweetDate").index()});
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);

        int textIndex = data.attribute("TweetText").index();

        // Buklea alderantziz zeharkatzen da instantziak ezabatzeak bug-rik ez sortzeko
        for (int i = data.numInstances() - 1; i >= 0; i--) {
            Instance unekoa = data.instance(i);

            String tweetGarbia = cleanTweet(unekoa.stringValue(textIndex));
            if (tweetGarbia.isEmpty()) {
                data.remove(i);
                ezabatutakoakCount++;
            } else {
                unekoa.setValue(textIndex, tweetGarbia);
            }
        }

        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(path));
        saver.setInstances(data);
        saver.writeBatch();

        System.out.println(path + " fitxategiko datuak aurreprozesatu dira. "
                + ezabatutakoakCount + " ezabatu dira hutsik zeudelako.");
    }

    /**
     * Tweet baten testua garbitu eta aurreprozesatzen du sailkapenerako.
     * @param tweet Garbitu nahi den Tweet-aren jatorrizko String-a
     * @return Testu normalizatua, karaktere berezirik gabe eta tokenizaziorako prest.
     */
    public static String cleanTweet(String tweet){
        tweet = tweet.replaceAll("http\\S+"," ");
        tweet = tweet.replaceAll("@\\w+\\s?"," ");
        tweet = tweet.replace("#"," ");
        tweet = tweet.replaceAll("[^a-zA-ZáéíóúÁÉÍÓÚñÑ\\s]", "");
        tweet = tweet.replaceAll("\\s+", " ").trim();
        return tweet;
    }
}