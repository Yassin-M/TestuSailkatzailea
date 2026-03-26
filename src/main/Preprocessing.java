package main;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

public class Preprocessing {

    public static void tweetakGarbitu(String arg) throws Exception{
        String path = arg;
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
     * String baten garbiketa egingo du karaktere
     * @param tweet Garbituko den Tweet-a
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
