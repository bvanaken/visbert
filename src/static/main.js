var ANERCorp_test = "./static/samples/ANERCorp_test.json";
var NERCorp_test = "./static/samples/NERCorp_test.json";
var NewsWire_test = "./static/samples/NewsWire_test.json";
var Wikipedia_test = "./static/samples/Wikipedia_test.json";

var color = Chart.helpers.color;
var fontDefault = {
    size: 13,
    weight: 'normal',

};
var fontBold = {
    size: 13,
    weight: 'bold'
};
var fontHighlighted = {
    size: 16,
    weight: 'bold'
};

var scatterChartData = {
    datasets: [
        {
            label: 'LOC',
            backgroundColor: color('green').alpha(0.2).rgbString(),
            data: [],
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'ORG',
            data: [],
            backgroundColor: color('darkcyan').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'PERS',
            data: [],
            backgroundColor: color('deepskyblue').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'MISC',
            data: [],
            backgroundColor: color('palevioletred').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'OTHER',
            data: [],
            backgroundColor: color('saddlebrown').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'IGNORE',
            data: [],
            backgroundColor: color('grey').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'Focus',
            data: [],
            backgroundColor: color('purple').rgbString(),
            datalabels: {
                font: fontHighlighted
            }
        }
    ]
};

var scatterChartHeadData = {
    datasets: [
        {
            label: 'LOC',
            backgroundColor: color('green').alpha(0.2).rgbString(),
            data: [],
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'ORG',
            data: [],
            backgroundColor: color('darkcyan').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'PERS',
            data: [],
            backgroundColor: color('deepskyblue').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'MISC',
            data: [],
            backgroundColor: color('palevioletred').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'OTHER',
            data: [],
            backgroundColor: color('saddlebrown').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'IGNORE',
            data: [],
            backgroundColor: color('grey').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'Focus',
            data: [],
            backgroundColor: color('purple').rgbString(),
            datalabels: {
                font: fontHighlighted
            }
        }
    ]
};

var phase1 = "Phase 1: Topical / Word Clusters";
var phase2 = "Phase 2: Entity Relation Clusters";
var phase3 = "Phase 3: Matching Supporting Facts with Question";
var phase4 = "Phase 4: Answer Extraction";
var color_map = {"B-LOC": "green", "I-LOC":"green", "B-PERS":"deepskyblue","I-PERS":"deepskyblue","B-PER":"deepskyblue","I-PER":"deepskyblue", "B-ORG":"darkcyan","I-ORG":"darkcyan","B-MISC":"palevioletred","I-MISC":"palevioletred","O":"saddlebrown"};
var hiddenStates = [];
var annotated_hiddenStates = [];
var attention_head = [];
var attention_heat;
var change_heatmap;
var predictedAnswer = [];
var focus_data = [];
var scatterPlot;
var scatterHeadPlot;
var tokenInfo = {};
var currentTask = 'model1';
var currentLayer = 0;
var currentAttentionLayer = 0;
var currentHeadLayer = 0;
var currentAttentionHead = 0;
var currentHead = 0;
var ownExample = false;
var predictedAnswer = [];
var mistakes = [];
var similarity_data = [];
var head_similarity_data;
var embedding_similarity = [];

var basePhaseLabels = {
    0: phase1,
    1: phase1,
    2: phase1,
    3: phase1 + " → " + phase2,
    4: phase2,
    5: phase2,
    6: phase2 + " → " + phase3,
    7: phase3,
    8: phase3,
    9: phase3 + " → " + phase4,
    10: phase4,
    11: phase4,
    12: phase4
};

var data_labels_map =  {
    ANERCorp:{
        file: ANERCorp_test,
        ner_labels: ['B-LOC', 'O', 'B-ORG', 'I-ORG', 'B-PERS', 'I-PERS', 'I-LOC', 'B-MISC', 'I-MISC']
    },
    NERCorp:{
        file: NERCorp_test,
        ner_labels: ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
    },

};

var tasks = {
    model1: {
        file: ANERCorp_test,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        attention_layer_nr: 11,
        attention_head_nr: 11,
        phaseLabels: basePhaseLabels,
        ner_labels: ['B-LOC', 'O', 'B-ORG', 'I-ORG', 'B-PERS', 'I-PERS', 'I-LOC', 'B-MISC', 'I-MISC']
    },
    model2: {
        file: ANERCorp_test,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        attention_layer_nr: 11,
        attention_head_nr: 11,
        phaseLabels: basePhaseLabels,
        ner_labels: ['B-LOC', 'O', 'B-ORG', 'I-ORG', 'B-PERS', 'I-PERS', 'I-LOC', 'B-MISC', 'I-MISC']
    },
    model3: {
        file: ANERCorp_test,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        attention_layer_nr: 11,
        attention_head_nr: 11,
        phaseLabels: basePhaseLabels,
        ner_labels: ['B-LOC', 'O', 'B-ORG', 'I-ORG', 'B-PERS', 'I-PERS', 'I-LOC', 'B-MISC', 'I-MISC']
    },
    model4: {
        file: ANERCorp_test,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        attention_layer_nr: 11,
        attention_head_nr: 11,
        phaseLabels: basePhaseLabels,
        ner_labels: ['B-LOC', 'O', 'B-ORG', 'I-ORG', 'B-PERS', 'I-PERS', 'I-LOC', 'B-MISC', 'I-MISC']
    },
    model5: {
        file: NERCorp_test,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        attention_layer_nr: 11,
        attention_head_nr: 11,
        phaseLabels: basePhaseLabels,
        ner_labels: ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
    }
};

function findMinMax(data) {
    var minX = 1000000000;
    var maxX = -1000000000;
    var minY = 1000000000;
    var maxY = -1000000000;

    for (let i = 0; i < data.length; i++) {
        var point = data[i];
        if (point.x < minX) {
            minX = point.x
        }
        if (point.x > maxX) {
            maxX = point.x
        }
        if (point.y < minY) {
            minY = point.y
        }
        if (point.y > maxY) {
            maxY = point.y
        }
    }
    return [minX, maxX, minY, maxY]
}

function updateZoomAndPan(minX, maxX, minY, maxY) {
    var xAxis = scatterPlot.options.scales.xAxes[0];
    var yAxis = scatterPlot.options.scales.yAxes[0];

    xAxis.ticks.min = minX;
    xAxis.ticks.max = maxX;
    yAxis.ticks.min = minY;
    yAxis.ticks.max = maxY;

    scatterPlot.options.pan.rangeMin = {x: minX, y: minY};
    scatterPlot.options.pan.rangeMax = {x: maxX, y: maxY};

    scatterPlot.options.zoom.rangeMin = {x: minX, y: minY};
    scatterPlot.options.zoom.rangeMax = {x: maxX, y: maxY};
}

function updateHeadZoomAndPan(minX, maxX, minY, maxY) {
    var xAxis = scatterHeadPlot.options.scales.xAxes[0];
    var yAxis = scatterHeadPlot.options.scales.yAxes[0];

    xAxis.ticks.min = minX;
    xAxis.ticks.max = maxX;
    yAxis.ticks.min = minY;
    yAxis.ticks.max = maxY;

    scatterHeadPlot.options.pan.rangeMin = {x: minX, y: minY};
    scatterHeadPlot.options.pan.rangeMax = {x: maxX, y: maxY};

    scatterHeadPlot.options.zoom.rangeMin = {x: minX, y: minY};
    scatterHeadPlot.options.zoom.rangeMax = {x: maxX, y: maxY};
}

// Refresh ScatterPlot

function refreshData(newData, newAnnotatedData, scatterPlot) {

    var minMaxPoints = findMinMax(newData);

    // add padding to avoid clipping labels
    minMaxPoints[0] -= 1;
    minMaxPoints[1] += 3;
    minMaxPoints[2] -= 1;
    minMaxPoints[3] += 2;

    updateZoomAndPan(minMaxPoints[0], minMaxPoints[1], minMaxPoints[2], minMaxPoints[3]);

    scatterChartData.datasets[0].data = newAnnotatedData.loc_points;
    scatterChartData.datasets[1].data = newAnnotatedData.org_points;
    scatterChartData.datasets[2].data = newAnnotatedData.pers_points;
    scatterChartData.datasets[3].data = newAnnotatedData.misc_points;
    scatterChartData.datasets[4].data = newAnnotatedData.o_points;
    scatterChartData.datasets[5].data = newAnnotatedData.ignore_points;
    scatterChartData.datasets[6].data = newAnnotatedData.focus_points;

    scatterPlot.update();
}

function refreshHeadData(newData, newAnnotatedData, scatterPlot) {

    var minMaxPoints = findMinMax(newData);

    // add padding to avoid clipping labels to keep the labels in the plot
    minMaxPoints[0] -= 0.01;
    minMaxPoints[1] += 0.03;
    minMaxPoints[2] -= 0.01;
    minMaxPoints[3] += 0.02;

    updateHeadZoomAndPan(minMaxPoints[0], minMaxPoints[1], minMaxPoints[2], minMaxPoints[3]);

    scatterChartHeadData.datasets[0].data = newAnnotatedData.loc_points;
    scatterChartHeadData.datasets[1].data = newAnnotatedData.org_points;
    scatterChartHeadData.datasets[2].data = newAnnotatedData.pers_points;
    scatterChartHeadData.datasets[3].data = newAnnotatedData.misc_points;
    scatterChartHeadData.datasets[4].data = newAnnotatedData.o_points;
    scatterChartHeadData.datasets[5].data = newAnnotatedData.ignore_points;
    scatterChartHeadData.datasets[6].data = newAnnotatedData.focus_points;

    scatterPlot.update();
}

//Refresh BarchartPlot

function plot_logits(logits, predictionPlot){
        predictionPlot.data.labels = logits.labels;
        predictionPlot.data.datasets[0].data = logits.values;
        var background = []
        for (element of logits.labels){
               background.push(color_map[element])
          }
        predictionPlot.data.datasets[0].backgroundColor = background
        predictionPlot.update();
}

function compute_focus_logits(focus_data, layerByLayerPlot){
        layerByLayerPlot.data.labels = focus_data.labels;
        layerByLayerPlot.data.datasets[0].data = focus_data.focus_prediction;
        var background = []
        for (element of focus_data.labels){
               background.push(color_map[element])
          }
        layerByLayerPlot.data.datasets[0].backgroundColor = background
        layerByLayerPlot.update()
}

function compute_embedding_similarity(embedding_similarity, embeddingSimilarityPlot){
        embeddingSimilarityPlot.data.labels = embedding_similarity.labels;
        embeddingSimilarityPlot.data.datasets[0].data = embedding_similarity.similarity;
        var background = []
        for (element of embedding_similarity.labels){
               background.push('DarkOliveGreen')
          }
        embeddingSimilarityPlot.data.datasets[0].backgroundColor = background
        embeddingSimilarityPlot.update();
}

function generate_attention_summary(data, Plot1, Plot2, color1, color2, label1, label2, title){
    Plot1.data.labels = data.labels
    Plot1.data.datasets[0].data = data.analysis1
    Plot1.data.datasets[0].label =  label1
    Plot1.options.title.text =  title
    Plot2.data.labels = data.labels
    Plot2.data.datasets[0].data = data.analysis2
    Plot2.data.datasets[0].label =  label2
    Plot2.options.title.text =  title
     var plot1_background = []
     var plot2_background = []
        for (element of data.labels){
               plot1_background.push(color1)
               plot2_background.push(color2)
          }
     Plot1.data.datasets[0].backgroundColor = plot1_background
     Plot2.data.datasets[0].backgroundColor = plot2_background

    Plot1.update()
    Plot2.update()

}

function compute_similarity(similarity_data, SimilarityPlot, measure = 'similarity'){
        SimilarityPlot.data.labels = similarity_data.labels;
        if (measure == 'similarity'){
            SimilarityPlot.data.datasets[0].data = similarity_data.similarity;
        }
        else{
            SimilarityPlot.data.datasets[0].data = similarity_data.distance;
        }

        var background = []
        for (element of similarity_data.labels){
               background.push(color_map[element.split('_')[2]])
          }
        SimilarityPlot.data.datasets[0].backgroundColor = background
        SimilarityPlot.update()
}

function compute_head_similarity(head_similarity_data, HeadSimilarityPlot, measure = 'similarity'){
        HeadSimilarityPlot.data.labels = head_similarity_data.labels;
        if (measure == 'similarity'){
            HeadSimilarityPlot.data.datasets[0].data = head_similarity_data.similarity;
        }
        else{
            HeadSimilarityPlot.data.datasets[0].data = head_similarity_data.distance;
        }

        var background = []
        for (element of head_similarity_data.labels){
               background.push(color_map[element.split('_')[2]])
          }
        HeadSimilarityPlot.data.datasets[0].backgroundColor = background
        HeadSimilarityPlot.update()
}

// Control Barchart View
function change_measure(){
    measure = parseText($('#similarity_measure'));
    compute_similarity(similarity_data[currentLayer], SimilarityPlot, measure)
    compute_head_similarity(head_similarity_data, HeadSimilarityPlot, measure)
}

function change_analysis(){

    var skillsSelect = document.getElementById("change_attention_analysis");
    var selectedText = skillsSelect.options[skillsSelect.selectedIndex].text;
    if (selectedText == 'produce_receive'){
        generate_attention_summary(attention_summary, Plot1, Plot2, 'palevioletred', 'saddlebrown', 'Produce', 'Receive', 'The amount of attention each prediction token produce/receive');
     }
     else{
        generate_attention_summary(local_global, Plot1, Plot2, 'darkcyan', 'deepskyblue', 'Local', 'Global', 'The amount of local/global attention each prediction token produce');
     }
}

function change_plot_view(){

    var skillsSelect = document.getElementById("choose_view");
    var selectedText = skillsSelect.options[skillsSelect.selectedIndex].text;
    if (selectedText == 'attention_head'){
        $("#attention_heat").html(attention_heat);
     }
     else{
        $("#attention_heat").html(change_heatmap);
     }
}

function reset_plots(){
        predictionPlot.data.labels = [];
        predictionPlot.data.datasets[0].data = [];
        layerByLayerPlot.data.labels = [];
        layerByLayerPlot.data.datasets[0].data = [];
        scatterChartData.datasets[0].data = []
        scatterChartData.datasets[1].data = []
        scatterChartData.datasets[2].data = []
        scatterChartData.datasets[3].data = []
        scatterChartData.datasets[4].data = []
        scatterChartData.datasets[5].data = []
        scatterChartData.datasets[6].data = []

        scatterChartHeadData.datasets[0].data = []
        scatterChartHeadData.datasets[1].data = []
        scatterChartHeadData.datasets[2].data = []
        scatterChartHeadData.datasets[3].data = []
        scatterChartHeadData.datasets[4].data = []
        scatterChartHeadData.datasets[5].data = []
        scatterChartHeadData.datasets[6].data = []


        SimilarityPlot.data.labels = [];
        SimilarityPlot.data.datasets[0].data = [];
        Plot1.data.labels = [];
        Plot1.data.datasets[0].data = [];
        Plot2.data.labels = [];
        Plot2.data.datasets[0].data = [];
        HeadSimilarityPlot.data.labels = [];
        HeadSimilarityPlot.data.datasets[0].data = [];
        $("#attention_heat").html('')

        $("#layer-nr-slider").val(0);
        currentLayer = 0
        refreshLayerNr(currentLayer);
        $("#attention_layer").val(0);
        currentAttentionLayer = 0
        $("#attention_head").val(0);
        currentAttentionHead = 0
        refreshAttentionNr(currentAttentionLayer, currentAttentionHead);
        $("#attention_l").val(0);
        currentLayer = 0
        $("#attention_h").val(0);
        currentHead = 0
        refreshHeadNr(currentLayer, currentHead);



        predictionPlot.update()
        layerByLayerPlot.update()
        scatterPlot.update();
        scatterHeadPlot.update();
        SimilarityPlot.update();
        Plot1.update();
        Plot2.update();
        HeadSimilarityPlot.update();
}

// Sample Functions

function separate_sentence_labels(sentenceList){
        sentence = sentenceList.split('*$*')
        count = 0;
        var words = "";
        var gold_standard = "";
        var sentence_labels = "";
        var sentence_words = "";
        for (element of sentence){
                   words+= "Word ("+count + ") : " + element.split("&#&")[0]
                   words+=" # "
                   gold_standard+= "Word ("+count + ") : " + element.split("&#&")[1]
                   gold_standard+=" # "
                   count++;
                   sentence_labels+= "<span style='color:"+color_map[element.split("&#&")[1]]+"'>" +element.split("&#&")[0]+ "</span>";
                   sentence_labels+=" "
          }
        return {
            words,
            gold_standard,
            sentence_labels,
        };
}

function insertSample(sample) {

    $('#predicted-labels').val("");
    $('#mistakes').val("");

    $('#id-input').prop('disabled', false);
    $('#id-input').val(sample.sentence_number);
    let { words, gold_standard, sentence_labels} = separate_sentence_labels(sample.sentence_labels);
    $('#sentence').val(sample.sentence);
    $('#annotated_sentence').val(words);
    $('#gold_standard').val(gold_standard);
    document.getElementById("sentence_labels").innerHTML = sentence_labels;
    $('#sentence_labels').addClass('goldstandard-highlighted');
    initialize_dropdown(currentTask)

    var task = tasks[currentTask];
    if (task.currentIndex === 0) {
        $('#id-switcher-left').addClass("id-switcher-label-inactive");
    } else {
        $('#id-switcher-left').removeClass('id-switcher-label-inactive');
    }
    if (task.currentIndex >= task.samples.length - 1) {
        $('#id-switcher-right').addClass('id-switcher-label-inactive');
    } else {
        $('#id-switcher-right').removeClass('id-switcher-label-inactive');
    }
}

function removeSample() {
    $('#id-input').prop('disabled', true);
    $('#sentence').val("");
    $('#annotated_sentence').val("");
    $('#gold_standard').val("");
    $('#sentence_labels').text("");
    $('#sentence_labels').removeClass('goldstandard-highlighted');
    $('#predicted-labels').val("");
    $('#agreement').val("");
    $('mistakes').val("");
    removeOptions(document.getElementById('focus'));


    $('#id-switcher-right').addClass('id-switcher-label-inactive');
    $('#id-switcher-left').addClass('id-switcher-label-inactive');
}

// Dropdown Functions

function initialize_dropdown(model_name){
    removeOptions(document.getElementById('focus'));
    id = parseText($('#id-input'));
    sentence = parseText($('#sentence'));
    labels = parseText($('#gold_standard'));
    mode = parseText($('#mode'));
    var data = {};
    data.id = id;
    data.sentence = sentence;
    data.labels = labels;
    data.model_name = model_name;
    data.mode = mode;
    data.ner_labels = tasks[currentTask].ner_labels

    $.ajax({
        url: '/dropdown',
        type: 'post',
        data: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
        dataType: 'json',
        success: function (data) {
            populate_dropdown(data.annotated_tokens);
            populate_tabs(data);
            populate_agreement(data);
            populate_true_mistakes(data);
        },
        error: predictionError
    });

}

function change_dropdown(){
        initialize_dropdown(currentTask)
}
function populate_dropdown(options){
    var select = document.getElementById("focus");
    for(var i = 0; i < options.length; i++) {
        var opt = options[i];
        var el = document.createElement("option");
        el.textContent = opt;
        el.value = opt;
        select.appendChild(el);
    }
}

function populate_tabs(data){
    var tab1 = document.getElementById('model1Tab');
    var tab2 = document.getElementById('model2Tab');
    var tab3 = document.getElementById('model3Tab');
    var tab4 = document.getElementById('model4Tab');
    var tab5 = document.getElementById('model5Tab');
    tab1.innerHTML = data.tab1 + " <a href='https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/ay94/5e2cea01a94359a494cc120199bb5d98/raw/cf307e63ebbb40b526f45c1f4735326c1f389233/arabertv02_config.json'><span class='footnote-id'>[1]</span></a>";
    tab2.innerHTML = data.tab2 + " <span class='footnote-id'>[2]</span></div>";
    tab3.innerHTML = data.tab3 + " <span class='footnote-id'>[3]</span>";
    tab4.innerHTML = data.tab4 + " <span class='footnote-id'>[4]</span>";
    tab5.innerHTML = data.tab5 + " <span class='footnote-id'>[5]</span>";
}

function populate_agreement(data){

        agreement = data.agreement
        count = 0;
        var agreements = "";

        for (element of agreement){
                   agreements+= "Word ("+count + ") : " + element
                   agreements+=" # "
                   count++;
          }
       document.getElementById("agreement").innerHTML = agreements;
}

function populate_true_mistakes(data){

        mistakes = data.true_mistakes
        count = 0;
        var true_mistakes = "";
        true_mistakes+= "Number of True Mistakes: " +mistakes.length+ " => "
        for (element of mistakes){
                   true_mistakes+= "Word ("+count + ") : " + element
                   true_mistakes+=" # "
                   count++;
          }
       document.getElementById("true_mistakes").innerHTML = true_mistakes;
}

function removeOptions(selectElement) {
   var i, L = selectElement.options.length - 1;
   for(i = L; i >= 0; i--) {
      selectElement.remove(i);
   }
}

function parseText(component) {

    function trim(s) {
        return (s || '').replace(/^\s+|\s+$/g, '');
    }
    var text = component.val();
    text = encodeURIComponent(text);
    text = text.replace(/\n/g, " ");
    text = text.replace(/\"/g, "'");
    text = trim(text);
    return text;
}

function specify_prediction_mistakes(currentLayer, prediction, mistakes){
        predicted_labels = ''
        count = 0
        for (element of prediction[currentLayer].labels){
                   predicted_labels+= "Word ("+ count + ") : " + element
                   predicted_labels+=" # "
                   count++;
          }
          $('#predicted-labels').val(predicted_labels);
          $('#mistakes').val(mistakes[currentLayer]);

    }

function change_mode(){
        requestPredictionAndVis();
        requestAttention();
}

// Request Functions

function processResult(data) {

    mistakes = data.mistakes;
    predictedAnswer = data.predicted_tags;
    var tokens = data.tokens;
    hiddenStates = data.hidden_states;
    focus_data = data.prediction;
    similarity_data = data.layers_similarity;
    annotated_hiddenStates = data.annotated_hidden_states;
    attentionHeads = data.attentions_heads;
    annotated_attentionHeads = data.annotated_heads;

    var analysisSelect = document.getElementById("change_attention_analysis");
    var analysis = analysisSelect.options[analysisSelect.selectedIndex].text;
    var viewSelect = document.getElementById("choose_view");
    var view = viewSelect.options[viewSelect.selectedIndex].text;

    currentLayer = Math.min(tasks[currentTask].layer_nr, currentLayer);
    currentHeadLayer = Math.min(tasks[currentTask].attention_layer_nr, currentHeadLayer);
    currentHead = Math.min(tasks[currentTask].attention_head_nr, currentHead);
    specify_prediction_mistakes(currentLayer, predictedAnswer, mistakes)
    refreshLayerNr(currentLayer);
    refreshHeadNr(currentHeadLayer, currentHead);
    adjustSlider(tasks[currentTask].layer_nr);
    refreshData(hiddenStates[currentLayer], annotated_hiddenStates[currentLayer], scatterPlot);
    refreshHeadData(attentionHeads[currentHeadLayer][currentHead], annotated_attentionHeads[currentHeadLayer][currentHead], scatterHeadPlot);

     plot_logits(predictedAnswer[currentLayer], predictionPlot);
     compute_focus_logits(focus_data[currentLayer], layerByLayerPlot);
     compute_similarity(similarity_data[currentLayer], SimilarityPlot);

    $('#button-spinner').hide();
    $('#attention_button_spinner').hide();
    $('#impact_button_spinner').hide();
}

function requestPredictionAndVis() {

    $('#button-spinner').show();
    $('#attention_button_spinner').show();
    $('#impact_button_spinner').show();

    var sample = {};
    sample.id = parseText($('#id-input'));
    sample.sentence = parseText($('#sentence'));
    sample.sentence_labels = parseText($('#sentence_labels'));
    sample.focus = parseText($('#focus'));
    sample.gold_standard = parseText($('#gold_standard'));
    sample.random = parseText($('#random'));
    sample.mode = parseText($('#mode'));

    sample.attention_layer = currentAttentionLayer
    sample.attention_head = currentAttentionHead

    var data = {};
    data.sample = sample;
    data.model = currentTask;
    data.ner_labels = tasks[currentTask].ner_labels

    $.ajax({
        url: '/predict',
        type: 'post',
        data: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
        dataType: 'json',
        success: function (data) {
            processResult(data);
        },
        error: predictionError
    });
}

function predictionError() {
    var errorMessage = "\< Prediction not possible \>";
    $('#predicted-answer').text(errorMessage);

    $('#button-spinner').hide();
    $('#attention_button_spinner').hide();
    $('#impact_button_spinner').hide();
}

function processImpact(data) {

    var task = tasks[currentTask]
    change_heatmap = data.change_heatmap
    $("#attention_heat").html(change_heatmap);
    $('#button-spinner').hide();
    $('#attention_button_spinner').hide();
    $('#impact_button_spinner').hide();
    $("#choose_view").val("training_impact");
}

function requestImpact() {
    $('#button-spinner').show();
    $('#attention_button_spinner').show();
    $('#impact_button_spinner').show();

    var sample = {};
    sample.id = parseText($('#id-input'));
    sample.sentence = parseText($('#sentence'));
    sample.sentence_labels = parseText($('#sentence_labels'));
    sample.gold_standard = parseText($('#gold_standard'));
    sample.mode = parseText($('#mode'));

    var data = {};
    data.sample = sample;
    data.model = currentTask;
    data.ner_labels = tasks[currentTask].ner_labels

    $.ajax({
        url: '/impact',
        type: 'post',
        data: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
        dataType: 'json',
        success: function (data) {
            processImpact(data);
        },
        error: predictionError
    });
}

function processAttention(data) {

    attention_heat = data.attention_heat
    attention_summary = data.attention_summary
    local_global = data.local_global

    var analysisSelect = document.getElementById("change_attention_analysis");
    var analysis = analysisSelect.options[analysisSelect.selectedIndex].text;
    var viewSelect = document.getElementById("choose_view");
    var view = viewSelect.options[viewSelect.selectedIndex].text;
    head_similarity_data = data.head_similarity;
    compute_head_similarity(head_similarity_data, HeadSimilarityPlot)

    currentAttentionLayer = Math.min(tasks[currentTask].attention_layer_nr, currentAttentionLayer);
    currentAttentionHead = Math.min(tasks[currentTask].attention_head_nr, currentAttentionHead);
    refreshAttentionNr(currentAttentionLayer, currentAttentionHead);

    $("#attention_heat").html(attention_heat);
    $("#choose_view").val("attention_head");

     if (analysis == 'produce_receive'){
        generate_attention_summary(attention_summary, Plot1, Plot2, 'palevioletred', 'saddlebrown', 'Produce', 'Receive', 'The amount of attention each prediction token produce/receive');
     }
     else{
        generate_attention_summary(local_global, Plot1, Plot2, 'darkcyan', 'deepskyblue', 'Local', 'Global', 'The amount of local/global attention each prediction token produce');
     }

    $('#button-spinner').hide();
    $('#attention_button_spinner').hide();
    $('#impact_button_spinner').hide();


}

function requestAttention(){

    $('#button-spinner').show();
    $('#attention_button_spinner').show();
    $('#impact_button_spinner').show();

    var sample = {};
    sample.id = parseText($('#id-input'));
    sample.sentence = parseText($('#sentence'));
    sample.sentence_labels = parseText($('#sentence_labels'));
    sample.gold_standard = parseText($('#gold_standard'));
    sample.attention_layer = currentAttentionLayer
    sample.attention_head = currentAttentionHead
    sample.mode = parseText($('#mode'));
    sample.focus = parseText($('#focus'));

    var data = {};
    data.sample = sample;
    data.model = currentTask;
    data.ner_labels = tasks[currentTask].ner_labels

    $.ajax({
        url: '/attention',
        type: 'post',
        data: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
        dataType: 'json',
        success: function (data) {
            processAttention(data);
        },
        error: predictionError
    });
}

// Sample Related Functions
function getDataType() {
         var data = {};
         data.model_name = currentTask
         $.ajax({
            url: '/data_type',
            type: 'post',
            data: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            },
            dataType: 'json',
            success: function (data) {
                changeData(data);
            },
            error: predictionError
        });
}

function changeData(data) {
         var model_name = data.model_name;
         var data_name = data.data_name;
         tasks[model_name].file = data_labels_map[data_name].file
         tasks[model_name].ner_labels = data_labels_map[data_name].ner_labels
         loadSamples(tasks)
}

function loadSamples(tasks) {
    var task = tasks[currentTask];

    if (task.samples === null) {
        $.getJSON(task.file, function (json) {
            task.samples = json;
            task.currentIndex = 0
            insertSample(task.samples[task.currentIndex]);
        });
    } else {
        insertSample(task.samples[task.currentIndex]);
    }
}

function toggleOwnExample() {
    // toggle
    ownExample = !ownExample;

    if (ownExample) {
        removeSample();

    } else {
        loadSamples()
    }
}

function get_sample(){
    var task = tasks[currentTask];

    var sample_id = document.getElementById("id-input").value;

    if (task.currentIndex < task.samples.length - 1 && !ownExample){
            task.currentIndex = sample_id
            insertSample(task.samples[task.currentIndex]);
    }
}

function switchToModel1() {
    currentTask = 'model1';

    $('#model1Tab').addClass('task-tab-active');
    $('#model2Tab').removeClass('task-tab-active');
    $('#model3Tab').removeClass('task-tab-active');
    $('#model4Tab').removeClass('task-tab-active');
    $('#model5Tab').removeClass('task-tab-active');
    reset_plots();

    if (!ownExample) {
        getDataType();
    }
}

function switchToModel2() {
    currentTask = 'model2';

    $('#model1Tab').removeClass('task-tab-active');
    $('#model2Tab').addClass('task-tab-active');
    $('#model3Tab').removeClass('task-tab-active');
    $('#model4Tab').removeClass('task-tab-active');
    $('#model5Tab').removeClass('task-tab-active');
    reset_plots()
    if (!ownExample) {
        getDataType();
    }
}

function switchToModel3() {
    currentTask = 'model3';

    $('#model1Tab').removeClass('task-tab-active');
    $('#model2Tab').removeClass('task-tab-active');
    $('#model3Tab').addClass('task-tab-active');
    $('#model4Tab').removeClass('task-tab-active');
    $('#model5Tab').removeClass('task-tab-active');
    reset_plots();
    if (!ownExample) {
        getDataType();
    }
}

function switchToModel4() {
    currentTask = 'model4';

    $('#model1Tab').removeClass('task-tab-active');
    $('#model2Tab').removeClass('task-tab-active');
    $('#model3Tab').removeClass('task-tab-active');
    $('#model4Tab').addClass('task-tab-active');
    $('#model5Tab').removeClass('task-tab-active');
    reset_plots();
    if (!ownExample) {
        getDataType();
    }
}

function switchToModel5() {
    currentTask = 'model5';

    $('#model1Tab').removeClass('task-tab-active');
    $('#model2Tab').removeClass('task-tab-active');
    $('#model3Tab').removeClass('task-tab-active');
    $('#model4Tab').removeClass('task-tab-active');
    $('#model5Tab').addClass('task-tab-active');
    reset_plots();
    if (!ownExample) {
        getDataType();
    }
}

function switchIdLeft() {
    var task = tasks[currentTask];

    if (task.currentIndex > 0 && !ownExample) {

        task.currentIndex = task.currentIndex - 1;
        insertSample(task.samples[task.currentIndex]);
        reset_plots();
    }
}

function switchIdRight() {
    var task = tasks[currentTask];

    if (task.currentIndex < task.samples.length - 1 && !ownExample) {
        task.currentIndex = task.currentIndex + 1;
        insertSample(task.samples[task.currentIndex]);
        reset_plots();
    }
}

function adjustSlider(nr_layers) {
    $("#layer-nr-slider").attr("max", nr_layers);
    $("#layer_max").text(nr_layers);
}

function refreshLayerNr(newLayer) {
    currentLayer = newLayer;

    $("#phase-label").text(tasks[currentTask].phaseLabels[currentLayer]);
    $("#layer_nr_label").text("Layer " + currentLayer);
}

function refreshAttentionNr(newLayer, newHead){
    currentAttentionLayer = newLayer
    currentAttentionHead = newHead
    $("#attention_layer_nr_label").text("Layer " + newLayer);
    $("#attention_head_nr_label").text("Head " + newHead);
}

function refreshHeadNr(newLayer, newHead){
    currentHeadLayer = newLayer
    currentHead = newHead
    $("#attention_l_nr_label").text("Layer " + newLayer);
    $("#attention_h_nr_label").text("Head " + newHead);
}

$(document).ready(function () {

    $("#layer-nr-slider").val(0);
    $("#attention_layer").val(0);
    $("#attention_head").val(0);
    $("#attention_l").val(0);
    $("#attention_h").val(0);
    getDataType('model1')

    $("#button-spinner").hide();
    $('#attention_button_spinner').hide();
    $('#impact_button_spinner').hide();

    var ctx = $('#plot_layers');
    var aspectRatio = (($(window).width() - 3000) / (-3060)).toFixed(1);
    aspectRatio = Math.max(aspectRatio, 0.5);
    aspectRatio = Math.min(aspectRatio, 1);

    scatterPlot = Chart.Scatter(ctx, {
        data: scatterChartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: aspectRatio,
            pan: {
                enabled: true,
                mode: "xy",
                speed: 0.01,
                rangeMin: {
                    x: 0,
                    y: 0
                },
                rangeMax: {
                    x: 0,
                    y: 0
                }
            },
            zoom: {
                enabled: true,
                drag: false,
                mode: "xy",
                speed: 0.02,
                rangeMin: {
                    x: 0,
                    y: 0
                },
                rangeMax: {
                    x: 10,
                    y: 10
                }
            },
            scales: {
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: 'PC 1'
                    },
                    ticks: {
                        maxRotation: 0,
                        precision: 0,
                        min: 0,
                        max: 10,
                        callback: function (value, index, values) {
                            return Math.round(value);
                        }
                    }
                }],
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: 'PC 2'
                    },
                    ticks: {
                        maxRotation: 0,
                        precision: 0,
                        min: 0,
                        max: 10,
                        callback: function (value, index, values) {
                            return Math.round(value);
                        }

                    }
                }]
            },
            tooltips: {
                callbacks: {
                    label: function (tooltipItem, data) {
                        return data.datasets[tooltipItem.datasetIndex].data[tooltipItem.index].label;
                    }
                }
            },
            plugins: {
                datalabels: {
                    anchor: 'end',
                    align: 'right',
                    color: 'grey',
                    offset: 8,
                    padding: 0,
                    clamp: true
                },
                legend: {
                    display: true
                },
                title: false
            }
        }
    });

    var heads_ctx = $('#plot_heads');
    var aspectRatio = (($(window).width() - 3000) / (-3060)).toFixed(1);
    aspectRatio = Math.max(aspectRatio, 0.5);
    aspectRatio = Math.min(aspectRatio, 1);

    scatterHeadPlot = Chart.Scatter(heads_ctx, {
        data: scatterChartHeadData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: aspectRatio,
            pan: {
                enabled: true,
                mode: "xy",
                speed: 0.01,
                rangeMin: {
                    x: 0,
                    y: 0
                },
                rangeMax: {
                    x: 0,
                    y: 0
                }
            },
            zoom: {
                enabled: true,
                drag: false,
                mode: "xy",
                speed: 0.02,
                rangeMin: {
                    x: 0,
                    y: 0
                },
                rangeMax: {
                    x: 10,
                    y: 10
                }
            },
            scales: {
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: 'PC 1'
                    },
                    ticks: {
                        maxRotation: 0,
                        precision: 0,
                        min: 0,
                        max: 10,
                        callback: function (value, index, values) {
                            return Math.round(value);
                        }
                    }
                }],
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: 'PC 2'
                    },
                    ticks: {
                        maxRotation: 0,
                        precision: 0,
                        min: 0,
                        max: 10,
                        callback: function (value, index, values) {
                            return Math.round(value);
                        }

                    }
                }]
            },
            tooltips: {
                callbacks: {
                    label: function (tooltipItem, data) {
                        return data.datasets[tooltipItem.datasetIndex].data[tooltipItem.index].label;
                    }
                }
            },
            plugins: {
                datalabels: {
                    anchor: 'end',
                    align: 'right',
                    color: 'grey',
                    offset: 8,
                    padding: 0,
                    clamp: true
                },
                legend: {
                    display: true
                },
                title: false
            }
        }
    });

    $("#layer-nr-slider").on("input change", function () {
        var newLayer = $(this).val();

        if (newLayer !== currentLayer) {
            refreshLayerNr(newLayer);

            if (annotated_hiddenStates.length > currentLayer) {
                refreshData(hiddenStates[currentLayer], annotated_hiddenStates[currentLayer], scatterPlot);
                plot_logits(predictedAnswer[currentLayer], predictionPlot);
                compute_focus_logits(focus_data[currentLayer], layerByLayerPlot);
                compute_similarity(similarity_data[currentLayer], SimilarityPlot);
                specify_prediction_mistakes(currentLayer, predictedAnswer, mistakes)
            }
        }
    });

    $("#attention_layer").on("input change", function () {
        var newLayer = $(this).val();
        var newHead = $("#attention_head").val();

        if (newLayer !== currentAttentionLayer) {
            refreshAttentionNr(newLayer, newHead);
            requestAttention();
            compute_head_similarity(head_similarity_data, HeadSimilarityPlot)

        }
    });

    $("#attention_head").on("input change", function () {
        var newHead = $(this).val();
        var newLayer = $("#attention_layer").val();

        if (newHead !== currentAttentionHead) {
            refreshAttentionNr(newLayer, newHead);
            requestAttention();
            compute_head_similarity(head_similarity_data, HeadSimilarityPlot)

        }
    });

    $("#attention_l").on("input change", function () {
        var newLayer = $(this).val();
        var newHead = $("#attention_h").val();

        if (newLayer !== currentAttentionLayer) {
            refreshHeadNr(newLayer, newHead);
            refreshHeadData(attentionHeads[currentHeadLayer][currentHead], annotated_attentionHeads[currentHeadLayer][currentHead], scatterHeadPlot);


        }
    });

    $("#attention_h").on("input change", function () {
        var newHead = $(this).val();
        var newLayer = $("#attention_l").val();

        if (newHead !== currentAttentionHead) {
            refreshHeadNr(newLayer, newHead);
            refreshHeadData(attentionHeads[currentHeadLayer][currentHead], annotated_attentionHeads[currentHeadLayer][currentHead], scatterHeadPlot);

        }
    });

    $("#own-example-check").change(function () {
        toggleOwnExample();
    });

    var bar_ctx = $('#prediction');
    predictionPlot = Chart.Bar(bar_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "NER tags",
                backgroundColor: [],
                data: [],
            datalabels:{
            color:'red',
            anchor: 'end',
            align: 'top',
            font:'0.5',
            offset:10,
            display: false
            }

              }
            ]
          },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: 'Predictions of the current example'
            }
          }
    });

    var layer_logits_ctx = $('#focus_logits')
    layerByLayerPlot = Chart.Bar(layer_logits_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "NER tags",
                backgroundColor: [""],
                data: []
              }
            ],
           },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: 'Logits of the focus token'
            }
          }
    });

    var similarity_ctx = $('#similarity')
    SimilarityPlot = Chart.Bar(similarity_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "",
                backgroundColor: [],
                data: [],
                datalabels:{
                display: false
            }
              },
            ],
           },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: 'Similarity between focus and all tokens'
            }
          }
    });

    var similarity_ctx = $('#head_similarity')
    HeadSimilarityPlot = Chart.Bar(similarity_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "",
                backgroundColor: [],
                data: [],
                datalabels:{
                display: false
            }
              },
            ],
           },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: 'Head Similarity between focus and all tokens'
            }
          }
    });

    var plot1_ctx = $('#plot1')
    Plot1 = Chart.Bar(plot1_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "",
                backgroundColor: ['red'],
                data: [],
                datalabels:{
                display: false
            }
              },

            ],
           },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: ''
            }
          }
    });

    var plot2_ctx = $('#plot2')
    Plot2 = Chart.Bar(plot2_ctx, {
          type: 'bar',
          data: {
            labels: [],
            datasets: [
              {
                label: "",
                backgroundColor: ['red'],
                data: [],
                datalabels:{
                display: false
            }
              },

            ],
           },
          options: {
            legend: { display: true, responsive: true, },
            title: {
              display: true,
              text: ''
            }
          }
    });

});

