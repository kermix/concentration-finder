import colorlover as cl

def create_and_mix_color_scale(n_colors, color_scale=dict(size='12', type='qual', name='Paired')):
    cs_size, cs_type, cs_name = color_scale['size'], color_scale['type'], color_scale['name']
    colors_scale = cl.scales[cs_size][cs_type][cs_name]

    colors = cl.to_rgb(cl.interp(colors_scale, n_colors))

    n = n_colors // int(cs_size)
    mixed = []
    for i in range(n):
        mixed += colors[i::n]
    return [c for c in set(mixed)]


def rgb2hex(r, g, b):
    r,g,b = int(r), int(g), int(b)
    return "{:02x}{:02x}{:02x}".format(r,g,b)

def create_and_mix_color_scale_html(n_colors, color_scale=dict(size='12', type='qual', name='Paired')):
    cs_size, cs_type, cs_name = color_scale['size'], color_scale['type'], color_scale['name']
    colors_scale = cl.scales[cs_size][cs_type][cs_name]

    colors = cl.to_numeric(cl.to_rgb(cl.interp(colors_scale, n_colors)))

    n = n_colors // int(cs_size)
    mixed = []
    for i in range(n):
        mixed += colors[i::n]

    return [rgb2hex(r,g,b) for r,g,b in colors]