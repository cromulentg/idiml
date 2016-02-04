package com.idibon.ml.feature.skipgram

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature._
import org.json4s._

/** Skip-gram transformation
  *
  * Skip-grams are ProductFeatures generated by skipping up to k grams
  * between each of n grams from a sequence of features (ie grams).
  *
  * http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf
  *
  * @param k the maximum distance between each gram chosen
  * @param n the number of grams in a skip-gram
  */

class SkipgramTransformer(k: Int, n: Int) extends FeatureTransformer with Archivable[SkipgramTransformer, SkipgramTransformerLoader] {
  //a sequence of all permutations of possible skips
  val skipPermutations = getSkipPermutations()

 def apply(input: Seq[Feature[_]]): Seq[ProductFeature] = {

   //make input more efficient for random access
   val randomAccessInput = input match {
     case x: IndexedSeq[Feature[_]] => x
     case _ => input.toVector
   }

   val len = randomAccessInput.length

   if (len < math.min(n, k)) {
     return Seq[ProductFeature]()
   }
   else {
     val startGrams = (0 to len - n) //all the possible starting grams for a skip gram
     var result = List[ProductFeature]()

     startGrams.foreach { i => //for every possible starting feature
       skipPermutations.foreach { perm => //go through the permutations of skip positions
         if (perm.sum+n+i <= len) { //check if going to run out of bounds

           //loop through the skip positions, incrementing an index to
           //select each gram and append it to the skipgram
           //var skipgram = List[Feature[_]](input(i))
           var index = i
           val skipgram = input(i) +: perm.map{ j =>
             index = index+j+1
             input(index)
           }
           result = new ProductFeature(skipgram) :: result
         }
       }
     }
     result.toSeq
   }
 }

  /** Generates the sequence of all possible length n permutations of k (repeated).
    * This represents all the different skip orderings for this configration.
    *
    * @return A sequence of skip length permutations, normalized
    * with listToSums() to provide actual indices within a sequence that will be
    * skipped.
    */
  def getSkipPermutations(): Seq[Seq[Int]] = {
    Seq.fill(n - 1)((0 to k).toIndexedSeq)
      .flatten.combinations(n - 1)
      .map(e => e.permutations)
      .flatten
      .toSeq
  }

  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(JObject(List(
      JField("min", JInt(this.k)),
      JField("max", JInt(this.n)))))
  }
}

/** Paired loader class for SkipgramTransformer instances. */
class SkipgramTransformerLoader extends ArchiveLoader[SkipgramTransformer] {

  def load(engine: Engine, reader: Alloy.Reader, config: Option[JObject]) = {
    implicit val formats = DefaultFormats

    val skipgramConfig = config.map(_.extract[SkipgramConfig])

    // by default, generate all bi-grams with skip=2
    new SkipgramTransformer(skipgramConfig.flatMap(_.k).getOrElse(2),
      skipgramConfig.flatMap(_.n).getOrElse(2))
  }
}

/** JSON configuration data for the Skip-gram transform
  *
  * @param k - Maximum distance between tokens in a sequence used to
  * generate skipgrams (default: 2)
  * @param n - Number of tokens in a skipgram (default: 2)
  */
sealed case class SkipgramConfig(k: Option[Int], n: Option[Int])