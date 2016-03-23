package com.idibon.ml.feature.stopwords


import java.io.File

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.language.LanguageCode
import scala.io.Source

/**
  * Created by nick on 3/18/16.
  */
class StopwordsTransformer extends com.idibon.ml.feature.FeatureTransformer {

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  

  val stopFile = "/Users/nick/github/idibin/fixtures/stopwords/en.txt"
  val englishStopSet = Source.fromFile(stopFile).getLines().toList.toSet
  val stopwordMap = Map("eng" -> englishStopSet)

  def filter(x: Word, stops: Set[String]) = {
    !stops.contains(x.word.toLowerCase)
  }

  def apply(input: Seq[Word], lc: Feature[LanguageCode]): Seq[Word] = {
    val stopSet = stopwordMap.getOrElse(lc.get.iso_639_2.get, Set())
    input.filter(x => filter(x, stopSet))
  }

  def loadStopWords(dir: String) = {
    // load files

    // for each file:
     // create stop words set
     // create 3 letter language code

    // create map
  }

}
