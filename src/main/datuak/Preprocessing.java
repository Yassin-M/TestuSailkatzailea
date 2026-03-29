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
     * * Hurrengo pausoak egiten ditu:
     * <ul>
     *     <li>Fitxategia kargatu eta 'TweetId' eta 'TweetDate' atributuak ezabatu.</li>
     *     <li>Instantzia bakoitzeko testua {@link #cleanTweet(String)} bidez garbitu.</li>
     *     <li>Garbiketaren ostean hutsik geratzen diren instantziak datu-sortatik kendu.</li>
     *     <li>Emaitza jatorrizko fitxategi bidean (path) gainidatzi.</li>
     * </ul>
     * @param path Fitxategiaren helbide erlatibo edo absolutua (path).
     * @throws Exception Fitxategia kargatzean, gordetzean edo iragaztean akatsen bat gertatuz gero.
     */
    public static void tweetakGarbitu(String path) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();

        Remove rm = new Remove();
        rm.setAttributeIndicesArray(new int[]{data.attribute("TweetId").index(), data.attribute("TweetDate").index()});
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);

        int textIndex = data.attribute("TweetText").index();

        for(int i = 0; i<data.numInstances(); i++){
            Instance unekoa = data.instance(i);

            String tweetGarbia = cleanTweet(unekoa.stringValue(textIndex));
            if(tweetGarbia.isEmpty()){
                data.remove(i);
            }else{
                unekoa.setValue(textIndex, tweetGarbia);
            }
        }

        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(path));
        saver.setInstances(data);
        saver.writeBatch();
    }

    /**
     * Tweet baten testua garbitu eta aurreprozesatzen du sailkapenerako.
     * * Hurrengo ekintzak burutzen ditu:
     * <ul>
     *     <li>URLak ezabatzen ditu (http...).</li>
     *     <li>Erabiltzaileen aipamenak (@erabiltzailea) kentzen ditu.</li>
     *     <li>Hashtag ikurra (#) ezabatzen du, hitza mantenduz.</li>
     *     <li>Karaktere ez-alfabetiko guztiak iragazten ditu (zenbakiak eta ikurrak).</li>
     *     <li>Zuriune bikoitzak ezabatu eta testua trimmatzen du.</li>
     * </ul>
     * @param tweet Garbitu nahi den Tweet-aren jatorrizko String-a
     * @return Testu normalizatua, karaktere berezirik gabe eta tokenizaziorako prest.
     */
    public static String cleanTweet(String tweet){
        //tweet-ak garbitu
        //url-ak ezabatu
        tweet = tweet.replaceAll("http\\S+"," ");
        //beste erabiltzaileen aipamenak @ kendu
        tweet = tweet.replaceAll("@\\w+\\s?"," ");
        //# karaktereak kendu
        tweet = tweet.replaceAll("#"," ");
        //# lotuta duten hitzak banatu
        //tweet = tweet.replaceAll("([a-z])([A-Z])", "$1 $2");

        tweet = tweet.replaceAll("[^a-zA-ZáéíóúÁÉÍÓÚñÑ\\s]", "");
        tweet = tweet.replaceAll("\\s+", " ").trim();
        return tweet;
    }
}
