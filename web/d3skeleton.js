// Relational Skeleton
// r by is_entity (radius of a circle)
// color by item_class
d3.json(document.currentScript.getAttribute('json'), function (error, graph) {
    if (error) throw error;

    var width = 960;//parseInt(document.currentScript.getAttribute('width'));
    var height = 500;//parseInt(document.currentScript.getAttribute('height'));

    var color = d3.scale.category20().domain([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);

    var force = d3.layout.force()
        .charge(-120)
        .linkDistance(30)
        .size([width, height]);

    var svg = d3.select("#area_skeleton").append("svg")
        .attr("width", width)
        .attr("height", height);


    force.nodes(graph.nodes)
        .links(graph.links)
        .start();

    var link = svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .style("stroke-width", function (d) {
            return Math.sqrt(d.value);
        });

    var node = svg.selectAll(".node")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", function (d) {
            return d.is_entity ? 10 : 3;
        })
        .style("fill", function (d) {
            return d.no_fill ? "transparent" : color(d.item_class);
            // return color(d.item_class);  // d.group
        })
        .call(force.drag);


    node.append("title")
        .text(function (d) {
            return d.name;
        });

    force.on("tick", function () {
        link.attr("x1", function (d) {
            return d.source.x;
        })
            .attr("y1", function (d) {
                return d.source.y;
            })
            .attr("x2", function (d) {
                return d.target.x;
            })
            .attr("y2", function (d) {
                return d.target.y;
            });

        node.attr("cx", function (d) {
            return d.x;
        })
            .attr("cy", function (d) {
                return d.y;
            });
    });
});