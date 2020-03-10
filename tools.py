import colorlover as cl

def create_and_mix_color_scale(n_colors, color_scale=dict(size='12', type='qual', name='Paired')):
    cs_size, cs_type, cs_name = color_scale['size'], color_scale['type'], color_scale['name']
    colors_scale = cl.scales[cs_size][cs_type][cs_name]

    colors = cl.to_rgb(cl.interp(colors_scale, n_colors))

    half_index = len(colors)//2

    h1, h2 = colors[:half_index], colors[half_index::][::-1]

    for i, h2i in enumerate(h2):
        h1 = h1[:i * 2] + [h2i] + h1[i * 2:]

    return h1
