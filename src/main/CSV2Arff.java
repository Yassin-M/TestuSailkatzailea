package main;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;
/**
 * CSV fitxategiak garbitu eta Weka-rako ARFF formatura bihurtzen dituen klasea.
 *
 */
public class CSV2Arff {
    /**
     * Programaren sarrera puntua. Fitxategiak sortaka prozesatzen ditu:
     * train, dev eta test_blind.
     *
     * @param args ppp
     * @throws Exception Weka-rekin edo fitxategiekin arazoren bat egonez gero.
     */
    public static void arffPasatu(String arg) throws Exception{
        String inputPath = arg;
        String cleanedPath = "data/clean/sortaGarbia.csv";
        String outputPath = "data/arff/sortaGarbia.arff";
        cleanCSV(inputPath, cleanedPath);

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(cleanedPath));
        loader.setNominalAttributes("1,2");
        loader.setStringAttributes("3,4,5");
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }
    /**
     * CSV fitxategi batetik erroreak ematen dituzten erregistroak garbitzen ditu
     * @param inputPath CSV fitxategiaren bidea
     * @param cleanPath Fitxategi garbiaren bidea
     */
    public static void cleanCSV(String inputPath, String cleanPath){
        int esperotutakoZutabeak = 0;
        int lerroEzabatuak = 0;
        int lerroTotalak = 0;

        try(BufferedReader br = new BufferedReader(new FileReader(inputPath, StandardCharsets.UTF_8));
            BufferedWriter bw = new BufferedWriter(new FileWriter(cleanPath, StandardCharsets.UTF_8))){
            String unekoLerroa;
            boolean lehenLerroa = true;
            while((unekoLerroa = br.readLine())!=null){
                lerroTotalak++;
                String trimmedLerroa = unekoLerroa.trim();

                if(trimmedLerroa.isEmpty()) continue;

                if(!hasBalancedQuotes(trimmedLerroa)){
                    lerroEzabatuak++;
                    continue;
                }

                if(!trimmedLerroa.endsWith("\"")){
                    lerroEzabatuak++;
                    continue;
                }

                String[] zutabeak = trimmedLerroa.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if(lehenLerroa){
                    esperotutakoZutabeak = zutabeak.length;
                    bw.write(reconstruirLinea(zutabeak));
                    bw.newLine();
                    lehenLerroa = false;
                }else{
                    if(zutabeak.length==esperotutakoZutabeak){
                        bw.write(reconstruirLinea(zutabeak));
                        bw.newLine();
                    }else{
                        lerroEzabatuak++;
                    }
                }
            }
            System.out.println("Prozesatutako lerroak: " + lerroTotalak);
            System.out.println("Ezabatutako lerroak: " + lerroEzabatuak);
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    /**
     * CSV erregistro baten komilla bikoitzak komilla sinpleetan bihurtu.
     * @param zutabeak Jasoko den erregistroaren bektorea. Bektorearen elementu bakoitza, erregistro horren atributu bat da.
     */
    private static String reconstruirLinea(String[] zutabeak) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < zutabeak.length; i++) {
            String gelaxka = zutabeak[i].trim();
            gelaxka = gelaxka.replace("“", "'")
                    .replace("”", "'")
                    .replace("‘", "'")
                    .replace("’", "'");

            gelaxka = gelaxka.replaceAll("[\\p{Cntrl}&&[^\r\n\t]]", "");

            if (gelaxka.startsWith("\"") && gelaxka.endsWith("\"") && gelaxka.length() >= 2) {
                gelaxka = gelaxka.substring(1, gelaxka.length() - 1);
            }

            gelaxka = gelaxka.replace("\"", "'");

            sb.append("\"").append(gelaxka).append("\"");

            if (i < zutabeak.length - 1) {
                sb.append(",");
            }
        }
        return sb.toString();
    }

    private static boolean hasBalancedQuotes(String line) {
        long count = line.chars().filter(ch -> ch == '"').count();
        return count % 2 == 0;
    }
}
