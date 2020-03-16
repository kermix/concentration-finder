if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = {
    open_details_on_btn_click: function (n_clicks1, n_clicks2) {
        n_clicks1= parseInt(n_clicks1)
        n_clicks2= parseInt(n_clicks2)

        if (n_clicks1 > 0 || n_clicks2  > 0 )
            return true
        return false
    }
}