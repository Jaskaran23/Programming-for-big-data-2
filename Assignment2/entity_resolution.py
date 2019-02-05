import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType,StringType,FloatType
from pyspark.sql.functions import udf
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import split, explode


# assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
#assert spark.version >= '2.3' # make sure we have Spark 2.3+

def token_stop(dfval):
    d_val = re.split(r'\W+', dfval)
    d_list=[]
    f_list=[]
    for d in d_val:
        d_val1 = d.lower()
        d_list.append(d_val1)
    return d_list

def jaccard_sim(list1,list2):
    s1 = set(list1)
    s2 = set(list2)
    #intersect = len(s1.intersection(s2))
    #uni = (len(s1)+len(s2))-intersect
    #return intersect/uni
    return float(len(s1.intersection(s2))/len(s1.union(s2)))




class EntityResolution:


    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()
        #self.df1.show()
        #self.df2.show()
        #print(type(self.stopWordsBC))



    def preprocessDF(self, df, cols):
        df =df.withColumn('joinkey1',fn.concat(cols[0],fn.lit(' '),cols[1]))
        #df.show(truncate=False)
        udf_func = udf(token_stop, ArrayType(StringType()))
        df = df.withColumn('joinkey1', udf_func(df.joinkey1))

        #stopwords = list(self.stopWordsBC)
        remover = StopWordsRemover(inputCol="joinkey1", outputCol="joinkey",stopWords=list(self.stopWordsBC))
        df_preprocess = remover.transform(df)
        return df_preprocess



    def filtering(self, df1, df2):

        df1.createOrReplaceTempView("df1")
        df1 = spark.sql('select id as id1, joinkey as joinkey1 from df1')
        #df1.show()

        df11 = df1.withColumn('key1',explode(df1['joinkey1']))
        df11.createOrReplaceTempView('df11')
        #df11.show()

        df_f1 = df11.filter(df11.key1 != '') # removing rows which contains no values
        #df_f1.show()
        df_f1.createOrReplaceTempView("df_f1")

        ndf1 = spark.sql('select id1 as id11,key1 from df_f1')
        ndf1.createOrReplaceTempView("ndf1")
        #ndf1.show()


        df2.createOrReplaceTempView("df2")
        df2 = spark.sql('select id as id2, joinkey as joinkey2 from df2')
        #df2.show()
        df22 = df2.withColumn('key2', explode(df2['joinkey2']))

        df_f2 = df22.filter(df22.key2 != '')
        df_f2.createOrReplaceTempView("df_f2")
        ndf5 = spark.sql('select id2 as id22, key2 from df_f2')
        #ndf5.show()

        df3 = ndf5.join(ndf1,ndf5.key2 == ndf1.key1)
        #df3.show()


        df4 = df3.select('id11','id22').distinct()
        #print(df4.count())
        #df4.show(truncate=False)

        df5 = df4.join(df1, df4.id11 == df1.id1)
        #df5.show()

        df6 = df5.join(df2, df5.id22 == df2.id2)
        #df6.show()

        df7 = df6.select('id1','id2','joinkey1','joinkey2')
        #df7.show(79,truncate=False)

        return df7





    def verification(self, candDF, threshold):
        udf_func1 = udf(jaccard_sim,FloatType())
        df_cand = candDF.withColumn('jaccard_similarity', udf_func1(candDF.joinkey1,candDF.joinkey2))
        #df_cand.show()
        df_cand1 = df_cand.filter(df_cand['jaccard_similarity'] >= threshold)
        #df_cand1.show()
        return df_cand1



    def evaluate(self, result, groundTruth):
        result1 = set(result)
        groundTruth1 = set(groundTruth)
        precision = len(result1.intersection(groundTruth1))/len(result1)
        #print precision

        recall = len(result1)/len(groundTruth1)
        #print recall

        FMeasure = (2 * precision * recall)/(precision + recall)
        return (precision, recall, FMeasure)

    def jaccardJoin(self, cols1, cols2, threshold):

        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)

        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)

        print ("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)

        print ("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF



    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('entity_resolution').getOrCreate()
    sc = spark.sparkContext

    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")


    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)


    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect() #here .rdd has been added to make this line compatible with spark installed on my laptop
    #print(result)

    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample") \
                          .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect() #here .rdd has been added to make this line compatible with spark installed on my laptop
    #print(groundTruth)         # if any error comes .rdd should be removed to make code work
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
