package com.idibon.ml.feature.stopwords


import java.io.File

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.language.LanguageCode
import scala.io.Source

/**
  * Created by nick gaylord on 3/18/16.
  *
  * Stopwords are frequent tokens, listed in separate language-specific files,
  * that model builders may want to exclude from being counted as unigram
  * features. This transformer takes a sequence of Words and returns another
  * sequence of Words filtered to exclude the contents of the stopword file
  * corresponding to the document's LanguageCode
  */
class StopwordsTransformer extends com.idibon.ml.feature.FeatureTransformer {
  val stopwordMap = generateStopFileMap()

  /**
    * Get the contents of the stopword file directory
    *
    * @param dir -- the directory containing the various stopword files
    * @return -- returns a list of filenames in dir
    */
  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  /**
    * Iterate over a list of stopword files and extract their contents,
    * normalizing filenames to three-letter ISO 639-2 language codes
    *
    * @return -- a map from language codes to language-specific stopword lists
    */
  def generateStopFileMap() = {
    val stopFileDirectory = getClass.getClassLoader.getResource("stopwords/").getPath()
    val stopFileList = getListOfFiles(stopFileDirectory)
    val stopwordLists = stopFileList.map(x => {
      val stopwordSet = Source.fromFile(x).getLines().toList.toSet
      val twoLetterCode = x.getName.split("\\.")(0)
      val threeLetterCode = LanguageCode.normalize(twoLetterCode)
      (threeLetterCode, stopwordSet)
    }).collect({ case (Some(threeLetterCode), sws) => (threeLetterCode, sws) })
    stopwordLists.toMap
  }

  /**
    * Word-by-word, identify as a stopword or not
    *
    * @param x -- an individual word
    * @param stops -- the stopword list corresponding to the document's language code
    * @return -- Boolean, 1 if the word is not in the stopword list, 0 if it is
    */
  def filter(x: Word, stops: Set[String]) = {
    !stops.contains(x.word.toLowerCase)
  }

  /**
    * Take a sequence of words (a document) and return a sequence of words
    * with stopwords removed.
    *
    * @param input -- a sequence of Words
    * @param lc -- the document's LanguageCode
    * @return -- a sequence of Words that is a (not necessarily proper) subset of input
    */
  def apply(input: Seq[Word], lc: Feature[LanguageCode]): Seq[Word] = {
    val code: LanguageCode = lc.get
    val stopSet = stopwordMap.getOrElse(code.iso_639_2.getOrElse(""), Set())
    input.filter(x => filter(x, stopSet))
  }

}
