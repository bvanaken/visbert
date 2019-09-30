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

var hiddenStates = [];
var scatterPlot;
var tokenInfo = {};
var currentLayer = 0;
var ownExample = false;


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

function predictionError() {
    var errorMessage = "\< Prediction not possible \>";
    $('#predicted-answer').text(errorMessage);

    $('#button-spinner').hide();
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