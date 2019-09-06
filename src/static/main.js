
var squadExampleFile = "./static/squad_examples.json";
var hotpotExampleFile = "./static/hotpot_examples.json";
var babiExampleFile = "./static/babi_examples.json";

var color = Chart.helpers.color;

var fontDefault = {
    size: 12,
    weight: 'normal',

};
var fontBold = {
    weight: 'bold'
};
var fontHighlighted = {
    size: 14,
    weight: 'bold'
};

var scatterChartData = {
    datasets: [
        {
            label: 'Context Tokens',
            backgroundColor: color('darkcyan').alpha(0.2).rgbString(),
            borderColor: color('darkcyan').alpha(0.6).rgbString(),
            data: [],
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'Supporting Facts',
            data: [],
            backgroundColor: color('darkcyan').rgbString(),
            datalabels: {
                font: fontDefault
            }
        },
        {
            label: 'Question',
            data: [],
            backgroundColor: color('blue').rgbString(),
            datalabels: {
                font: fontBold
            }
        },
        {
            label: 'Predicted Answer',
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
    18: phase3  + " → " + phase4,
    19: phase4,
    20: phase4,
    21: phase4,
    22: phase4,
    23: phase4,
    24: phase4
};

var hiddenStates = [];
var scatterPlot;
var tokenInfo = {};
var currentTask = 'squad';
var currentSups = null;
var currentLayer = 0;
var ownExample = false;

var tasks = {
    squad: {
        file: squadExampleFile,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        phaseLabels: basePhaseLabels
    },
    babi: {
        file: babiExampleFile,
        samples: null,
        currentIndex: 0,
        layer_nr: 12,
        phaseLabels: basePhaseLabels
    },
    hotpot: {
        file: hotpotExampleFile,
        samples: null,
        currentIndex: 0,
        layer_nr: 24,
        phaseLabels: largePhaseLabels
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

function adjustSlider(nr_layers) {
    $("#layer-nr-slider").attr("max", nr_layers);
    $("#layer_max").text(nr_layers);
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

function refreshData(newData, scatterPlot) {

    var minMaxPoints = findMinMax(newData);

    // add padding to avoid clipping labels
    minMaxPoints[0] -= 1;
    minMaxPoints[1] += 3;
    minMaxPoints[2] -= 1;
    minMaxPoints[3] += 2;

    updateZoomAndPan(minMaxPoints[0], minMaxPoints[1], minMaxPoints[2], minMaxPoints[3]);

    var questionIds = tokenInfo.token_indices.question;
    var answerIds = tokenInfo.token_indices.answer;
    var sups = tokenInfo.token_indices.sups;
    var supDatapoints = [];

    for (let i = 0; i < sups.length; i++) {
        var supIds = sups[i];
        var supTokens;
        if ((answerIds.start > supIds.start && answerIds.start < supIds.end)
            || (answerIds.end > supIds.start && answerIds.end < supIds.end)) {
            supTokens = newData.slice(supIds.start, answerIds.start).concat(newData.slice(answerIds.end, supIds.end));
        } else {
            supTokens = newData.slice(supIds.start, supIds.end);
        }
        supDatapoints = supDatapoints.concat(supTokens)
    }

    var questionDatapoints = newData.slice(questionIds.start, questionIds.end);
    var answerDatapoints = newData.slice(answerIds.start, answerIds.end);
    var contextDatapoints = newData.slice(0, tokenInfo.token_indices.question.start).concat(
        newData.slice(tokenInfo.token_indices.question.end, tokenInfo.token_indices.answer.start),
        newData.slice(tokenInfo.token_indices.answer.end, newData.length)
    );

    scatterChartData.datasets[0].data = contextDatapoints;
    scatterChartData.datasets[1].data = supDatapoints;
    scatterChartData.datasets[2].data = questionDatapoints;
    scatterChartData.datasets[3].data = answerDatapoints;


    scatterPlot.update();
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

function insertSample(sample) {
    $('#id-input').prop('disabled', false);
    $('#id-input').val(sample.id);
    $('#question-textarea').val(sample.question);
    $('#ground-truth-answer').val(sample.answer);
    $('#predicted-answer').val("");
    $('#context-textarea').val(sample.context);

    currentSups = sample.sup_ids;
}

function removeSample() {
    $('#id-input').prop('disabled', true);
    $('#question-textarea').val("");
    $('#ground-truth-answer').val("");
    $('#predicted-answer').val("");
    $('#context-textarea').val("");

    currentSups = null;
}

function parseText(component) {

    function trim(s) {
        return (s || '').replace(/^\s+|\s+$/g, '');
    }

    var text = component.val();
    text = encodeURIComponent(text);

    // Remove linebreaks from input
    text = text.replace(/\n/g, " ");

    // Remove quotes from input
    text = text.replace(/\"/g, "'");

    text = trim(text);

    return text;
}


function processResult(data) {
    var predictedAnswer = data.prediction.text;

    $('#predicted-answer').val(predictedAnswer);

    hiddenStates = data.hidden_states;
    tokenInfo.token_indices = data.token_indices;

    // clip to largest layer nr
    currentLayer = Math.min(tasks[currentTask].layer_nr, currentLayer);
    refreshLayerNr(currentLayer);
    adjustSlider(tasks[currentTask].layer_nr);
    refreshData(hiddenStates[currentLayer], scatterPlot);

    $('#button-spinner').hide();
}

function predictionError() {
    var errorMessage = "\< Prediction not possible \>";
    $('#predicted-answer').val(errorMessage);

    $('#button-spinner').hide();
}

function requestPredictionAndVis() {

    $('#button-spinner').show();

    var sample = {};

    sample.id = parseText($('#id-input'));
    sample.question = parseText($('#question-textarea'));
    sample.context = parseText($('#context-textarea'));
    sample.answer = parseText($('#ground-truth-answer'));
    sample.sup_ids = currentSups;

    var data = {};
    data.sample = sample;
    data.model = currentTask;

    $.ajax({
        url: '/visbert/predict',
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

function switchToHotpot() {
    currentTask = 'hotpot';

    $('#hotpotTab').addClass('task-tab-active');
    $('#squadTab').removeClass('task-tab-active');
    $('#babiTab').removeClass('task-tab-active');

    if (!ownExample) {
        loadSamples();
    }
}

function switchToSquad() {
    currentTask = 'squad';

    $('#squadTab').addClass('task-tab-active');
    $('#hotpotTab').removeClass('task-tab-active');
    $('#babiTab').removeClass('task-tab-active');

    if (!ownExample) {
        loadSamples();
    }
}

function switchToBabi() {
    currentTask = 'babi';

    $('#squadTab').removeClass('task-tab-active');
    $('#hotpotTab').removeClass('task-tab-active');
    $('#babiTab').addClass('task-tab-active');

    if (!ownExample) {
        loadSamples();
    }
}

function switchIdLeft() {
    var task = tasks[currentTask];

    if (task.currentIndex > 0 && !ownExample) {
        task.currentIndex = task.currentIndex - 1;

        insertSample(task.samples[task.currentIndex]);
    }
}

function switchIdRight() {
    var task = tasks[currentTask];

    if (task.currentIndex < task.samples.length - 1 && !ownExample) {
        task.currentIndex = task.currentIndex + 1;

        insertSample(task.samples[task.currentIndex]);
    }
}

function refreshLayerNr(newLayer) {
    currentLayer = newLayer;

    $("#phase-label").text(tasks[currentTask].phaseLabels[currentLayer]);
    $("#layer_nr_label").text("Layer " + currentLayer);
}

$(document).ready(function () {

    $("#layer-nr-slider").val(0);
    loadSamples('squad');

    var ctx = $('#plot_canvas');

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

            if (hiddenStates.length > currentLayer) {
                refreshData(hiddenStates[currentLayer], scatterPlot);
            }
        }
    });

    $("#own-example-check").change(function () {
        toggleOwnExample();
    });

    $("#button-spinner").hide();

});