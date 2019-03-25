/*
 * JavaScript file for the application to demonstrate
 * using the API
 */

// Create the namespace instance
let ns = {};

// Create the model instance
ns.model = (function() {
    'use strict';

    let $event_pump = $('body');

    // Return the API
    return {
        'valider': function(url) {
            let ajax_options = {
                type: 'GET',
                url: 'api/predict/'+ btoa(url).toString(),
                accepts: 'application/json',
                dataType: 'json'
            };
            $.ajax(ajax_options)
            .done(function(data) {

                $event_pump.trigger('model_validate_success', [data]);
            })
            .fail(function(xhr, textStatus, errorThrown) {
                $event_pump.trigger('model_error', [xhr, textStatus, errorThrown]);
            })
        }
    };
}());

// Create the view instance
ns.view = (function() {
    'use strict';

    let $url = $('#url');

    // return the API
    return {
      getScore: function(data) {

              // clear the table
              // did we get a people array?
              console.log(data)
              $('#score').html("<h1>La probabilité d'avoir un texte véridique est de "+data*100+" %</h1>")

          }
      }
}());

// Create the controller
ns.controller = (function(m, v) {
    'use strict';

    let model = m,
        view = v,
        $event_pump = $('body'),
        $url = $('#url');

        $('#valider').click(function(e) {
            let url = $url.val();
            e.preventDefault();
            model.valider(url);
    });
    // Get the data from the model after the controller is done initializing
    // Validate input
    function validate(url) {
        return url !== "";
    }
    // Handle the model events
    $event_pump.on('model_validate_success', function(e, data) {
        view.getScore(data);
    });
    $event_pump.on('model_error', function(e, xhr, textStatus, errorThrown) {
        let error_msg = textStatus + ': ' + errorThrown ;
        console.log(error_msg);
    })
}(ns.model, ns.view));
