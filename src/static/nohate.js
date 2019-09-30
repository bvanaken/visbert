var engExampleFile = "./static/eng_examples.json";

var scatterChartData = {
    datasets: [
        {
            label: 'Tokens',
            backgroundColor: color('darkcyan').rgbString(),
            data: [],
            datalabels: {
                font: fontDefault
            }
        }
    ]
};

var phase1 = "Phase 1: Topical / Word Clusters";
var phase2 = "Phase 2: Entity Relation Clusters";
var phase3 = "Phase 3: Matching Supporting Facts with Question";
var phase4 = "Phase 4: Answer Extraction";

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

var largePhaseLabels = {
    0: phase1,
    1: phase1,
    2: phase1,
    3: phase1,
    4: phase1,
    5: phase1,
    6: phase1 + " → " + phase2,
    7: phase2,
    8: phase2,
    9: phase2,
    10: phase2 + " → " + phase3,
    11: phase3,
    12: phase3,
    13: phase3,
    14: phase3,
    15: phase3,
    16: phase3,
    17: phase3,
    18: phase3 + " → " + phase4,
    19: phase4,
    20: phase4,
    21: phase4,
    22: phase4,
    23: phase4,
    24: phase4
};

var currentTask = 'eng_large';

var tasks = {
    eng_large: {
        file: engExampleFile,
        samples: null,
        currentIndex: 0,
        layer_nr: 24,
        phaseLabels: largePhaseLabels
    }
};

function refreshData(newData, scatterPlot) {

    var minMaxPoints = findMinMax(newData);

    // add padding to avoid clipping labels
    minMaxPoints[0] -= 1;
    minMaxPoints[1] += 3;
    minMaxPoints[2] -= 1;
    minMaxPoints[3] += 2;

    updateZoomAndPan(minMaxPoints[0], minMaxPoints[1], minMaxPoints[2], minMaxPoints[3]);

    scatterChartData.datasets[0].data = newData;

    scatterPlot.update();
}

function insertSample(sample) {
    var groundTruth = "";
    if(sample.label.toString() === "1"){
        groundTruth = "HATE";
    } else {
        groundTruth = "NO HATE";
    }

    $('#id-input').prop('disabled', false);
    $('#id-input').val(sample.id);
    $('#input-textarea').val(sample.text);
    $('#ground-truth').text(groundTruth);
    $('#predicted-answer').text("");
    $('#predicted-answer').removeClass('answer-highlighted');

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
    $('#input-textarea').val("");
    $('#predicted-answer').text("");
    $('#ground-truth').text("");
    $('#predicted-answer').removeClass('answer-highlighted');

    $('#id-switcher-right').addClass('id-switcher-label-inactive');
    $('#id-switcher-left').addClass('id-switcher-label-inactive');
}


function processResult(data) {
    var predictedLabel = data.prediction.label;
    var probability = "";

    if(predictedLabel.toString() === "1"){
        probability = data.prediction.probability;
    } else {
        probability = 1-parseFloat(data.prediction.probability);
    }

    $('#predicted-answer').text(probability.toFixed(2) + " HATE Probability");
    $('#predicted-answer').addClass('answer-highlighted');

    hiddenStates = data.hidden_states;

    // clip to largest layer nr
    currentLayer = Math.min(tasks[currentTask].layer_nr, currentLayer);
    refreshLayerNr(currentLayer);
    adjustSlider(tasks[currentTask].layer_nr);
    refreshData(hiddenStates[currentLayer], scatterPlot);

    $('#button-spinner').hide();
}

function requestPredictionAndVis() {

    $('#button-spinner').show();

    var sample = {};

    sample.id = parseText($('#id-input'));
    sample.text = parseText($('#input-textarea'));

    var data = {};
    data.sample = sample;
    data.model = currentTask;

    $.ajax({
        url: '/visbert/predict_nohate',
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

function loadSamples() {
    var task = tasks[currentTask];

    if (task.samples === null) {
        $.getJSON(task.file, function (json) {
            task.samples = json;
            task.currentIndex = Math.floor(Math.random() * task.samples.length);

            insertSample(task.samples[task.currentIndex]);
        });
    } else {
        insertSample(task.samples[task.currentIndex]);
    }
}


$(document).ready(function () {

    loadSamples('eng');

});