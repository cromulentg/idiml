package com.idibon.ml.unicode
/**
  * Created by haley on 3/25/16.
  */

/** Empty class for JRuby to have a way of accessing companion class
  *
  * com.idibon.ml.unicode.UnicodeConverter
  */
class UnicodeConverter {

}

/** Converts offset, length of spans in document content to relative
  * offset, length in UTF-16 (IdiML) or UTF-32/code points (Ididat).
  *
  */
object UnicodeConverter {

  /**
    * Converts a document span represented by length and offset (code points)
    * to the same span in UTF-16 in order to be Idiml compatible.
    *
    * @param content the document content
    * @param offset32 the offset the span starts at in this document (in code points)
    * @param length32 the length of this span (in code points)
    * @return a tuple of offset, length representing the same span in UTF-16
    */
  def toIdiml(content: String, offset32: Int, length32: Int): (Int, Int) = {
    /* Given the code points of this content, iterate through
     * them and sum the Character.charCount of each point.
     * In UTF-16 this will be 1 for bmp, and 2 for smp, as opposed
     * to 1 for both planes in code points.
     */
    val(offset,length) = putInBounds(content.codePointCount(0,content.length()), offset32, length32)

    val cp = content.codePoints()
    val iter = cp.iterator()

    //updates as we iterate through the cps
    var cpCount = 0

    //keep track of the utf-16 length and offset values
    var offset16 = 0
    var length16 = 0

    //loop through "offset" number of code points
    while(iter.hasNext && cpCount < offset){
      cpCount = cpCount + 1
      offset16 = offset16 + Character.charCount(iter.next())
    }

    //reset and loop through "length" number of code points
    cpCount = 0
    while(iter.hasNext && cpCount < length){
      cpCount = cpCount + 1
      length16 = length16 + Character.charCount(iter.next())
    }

    (offset16, length16)
  }

  /**
    *Convert prediction spans generated in IdiML from UTF-16 relative indexes to UTF-32
    *
    * @param content the document content
    * @param offset16 the offset the span starts at in this document utf-16
    * @param length16 the length of this span in utf-16
    * @return a tuple of offset, length representing the same span in UTF-32 (code points)
    */
  def toIdidat(content: String, offset16: Int, length16: Int): (Int, Int) ={
    val(offset,length) = putInBounds(content.length(), offset16, length16)

    //get the number of code points in the string up to the offset, and the number
    //of code points in the string from offset -> length.
    (content.codePointCount(0,offset), content.codePointCount(offset,offset+length))
  }

  /**
    * To avoid any index out of bound cases from the get go, converts offset
    * length into an inbounds offset and length pair
    *
    * @param contentLength of the document content in the same unicode as offset, length
    * @param offset the offset the span starts at in this document
    * @param length the length of this span
    * @return a tuple of offset, length representing the span within bounds of the original content
    */
  def putInBounds(contentLength: Int, offset: Int, length: Int): (Int, Int) = {
    //four cases that could end up with out of bounds issues:
    if(offset>contentLength){ //offset out of bounds, empty string
      (0,0)
    }
    else if(length<=0){ //invalid length, end up with the empty string
      (0,0)
    }
    else if(offset < 0){ //invalid offset, check again with updated values
      putInBounds(contentLength,0, length+offset)
    }
    else if(offset+length > contentLength){ //length too long, just take the tail
      (offset, contentLength-offset)
    }
    else{
      (offset, length)
    }
  }
}
