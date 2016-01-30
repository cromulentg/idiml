#!/usr/bin/env ruby
# Evaluate prediction output against known gold values stored
# as metadata stored in the test set
#
# Documents should be newline-delimited JSON files, like those
# used for ml-app predict, with a gold label name (i.e., the
# expected prediction) stored under the metadata.gold key
require 'csv'
require 'json'
require 'optparse'
require 'pp'

options = {}
# default all predictions as identical to the gold label for analysis
conflated = Hash.new { |h, l| h[l] = l }
cli = OptionParser.new do |cli|
  cli.on('-g', '--gold JSON', 'Gold documents') do |gold_file|
    options[:gold] = gold_file
  end
  cli.on('-p', '--predictions CSV', 'Predictions output') do |predictions_file|
    options[:predictions] = predictions_file
  end
  cli.on('-l', '--label_conflate PREDICT_LABEL,GOLD_LABEL', Array,
         'Treat PREDICT_LABEL equal to GOLD_LABEL for evaluation') do |predict, gold|
    conflated[predict] = gold
  end
  cli.on_tail('-h', '--help', 'Show help') do
    options[:help] = true
  end
end

argv = ARGV.dup
argv = [ '-h' ] if argv.empty?
cli.parse(argv)

if options[:help]
  puts cli
  exit
end

expected = Hash[File.open(options[:gold]).each_line
                 .map { |l| JSON.parse(l) }
                 .map { |json| [json['name'], json['metadata']['gold']] }]

actual = Hash[CSV.read(options[:predictions], headers: true).map do |row|
  # extract all of the label confidences, then choose the highest one
  labels = row.headers[2..-1].select { |c| !(c =~ /^features\[[^\]]+\]$/) }
  best = (labels.zip(labels.map { |l| row[l].to_f }).max_by { |v| v[1] })[0]
  [ row['Name'], best ]
end]

results = Hash.new { |h, l| h[l] = { total: 0, right: 0,
  false_negative: 0, false_positive: Hash.new { |h, l| h[l] = 0 } } }

actual.each do |document, predicted|
  gold = expected[document]
  predicted = conflated[predicted]
  results[gold][:total] += 1
  if gold == predicted
    results[gold][:right] += 1
  else
    results[predicted][:false_positive][:total] += 1
    results[predicted][:false_positive][gold] += 1
    results[gold][:false_negative] += 1
  end
end

# compute per-label f1, recall and precision
results.each do |label, values|
  results[label][:recall] = (results[label][:right].to_f / results[label][:total])
  results[label][:precision] = (results[label][:right].to_f / (results[label][:right] + results[label][:false_positive][:total]))

  results[label][:f1] = (2 * results[label][:precision] * results[label][:recall]) / (results[label][:precision] + results[label][:recall])
end

# and now macro-level stats
accuracy = results.reduce(0) { |sum, r| sum + r[1][:right] }.to_f / results.reduce(0) { |sum, r| sum + r[1][:total] }
macro_average = results.reduce(0.0) { |sum, r| sum + r[1][:f1] } / results.size

PP.pp results
puts "Accuracy: #{accuracy}"
puts "Macro f1: #{macro_average}"
