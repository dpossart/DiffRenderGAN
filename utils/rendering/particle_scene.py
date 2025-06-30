import pathlib

MESH_PATH = 'utils/rendering/'
SUPPORTED_MESH_EXTENSIONS = ["*.ply", "*.obj"]


def create_sensor(stddev):
    return f"""<sensor type="perspective">
                <string name="fov_axis" value="x"/>
                <float name="fov" value="40"/>
                <float name="principal_point_offset_x" value="0.000000"/>
                <float name="principal_point_offset_y" value="-0.000000"/>
                <float name="near_clip" value="0.100000"/>
                <float name="far_clip" value="100.000000"/>
                <transform name="to_world">
                    <rotate x="1" angle="90"/>
                    <rotate y="1" angle="270"/>
                    <rotate z="1" angle="0"/>
                    <translate value="0.000000 6.0000000 0.000000"/>
                </transform>

                <sampler type="stratified">
                    <integer name="sample_count" value="100"/>
                </sampler>

                <film type="hdrfilm">
                    <integer name="width" value="256"/>
                    <integer name="height" value="256"/>
                    <string name="pixel_format" value="rgb"/>
                    <string name="component_format" value="float32"/>
                    <boolean name="sample_border" value="true"/>
                    <rfilter type="gaussian">
                        <float name="stddev" value="{stddev}"/>
                    </rfilter >

                </film>
            </sensor>"""


def build_particle_scene(sample_params_dict, mesh_params_dict):
    """
    Build a scene configuration for rendering mesh images.

    Args:
        sample_params_dict (dict): Dictionary containing material properties for meshes.
        mesh_params_dict (dict): Dictionary containing mesh parameters.
        gaussdev (float): Gaussian deviation.

    Returns:
        tuple: A tuple containing the number of mesh meshes and the scene configuration.
    """
    d_particle_keys = ['p_base_color', 'p_sheen', 'p_spec_tint', 'p_metallic', 'p_specular', 'p_anisotropic',
                       'p_spec_trans',
                       'p_clearcoat_gloss', 'p_clearcoat', 'p_flatness', 'p_sheen_tint', 'p_roughness']

    d_stage_keys = ['s_base_color', 's_sheen', 's_spec_tint', 's_metallic', 's_specular', 's_anisotropic',
                    's_spec_trans',
                    's_clearcoat_gloss', 's_clearcoat', 's_flatness', 's_sheen_tint', 's_roughness']

    # Remove all differentiable params that are bigger than 0
    # Keep the rest for optimization
    for key in sample_params_dict:
        if key in d_stage_keys:
            if sample_params_dict[key] < 0:
                sample_params_dict[key] = 0
            else:
                d_stage_keys.remove(key)
        if key in d_particle_keys:
            if sample_params_dict[key] < 0:
                sample_params_dict[key] = 0
            else:
                d_particle_keys.remove(key)

    mesh_path = mesh_params_dict['mesh_path']
    mesh_scale = mesh_params_dict['mesh_scale']

    mesh_dir = [
        str(f) for extension in SUPPORTED_MESH_EXTENSIONS
        for f in pathlib.Path(mesh_path).glob(extension)
    ]

    emission1 = sample_params_dict['emission1']
    emission2 = sample_params_dict['emission2']
    print(f'{len(mesh_dir)} meshes found in {mesh_path}')

    particle_meshes = ""
    for id, mesh in enumerate(mesh_dir):
        particle_meshes += f"""                
                <bsdf type="principled" id="particle_{id + 1}_bsdf">
                   <rgb name="base_color" value="{sample_params_dict['s_base_color']}"/>
                   <float name="roughness" value="{sample_params_dict['s_roughness']}" />
                   <float name="anisotropic" value="{sample_params_dict['s_anisotropic']}" />
                   <float name="metallic" value="{sample_params_dict['s_metallic']}" />
                   <float name="spec_trans" value="{sample_params_dict['s_spec_trans']}" />
                   <float name="specular" value="{sample_params_dict['s_specular']}" />
                   <float name="spec_tint"  value="{sample_params_dict['s_spec_tint']}" />
                   <float name="sheen" value="{sample_params_dict['s_sheen']}" />
                   <float name="sheen_tint" value="{sample_params_dict['s_sheen_tint']}" />
                   <float name="flatness" value="{sample_params_dict['s_flatness']}" />
                   <float name="clearcoat" value="{sample_params_dict['s_clearcoat']}" />
                   <float name="clearcoat_gloss" value="{sample_params_dict['s_clearcoat_gloss']}" />
               </bsdf>

               <shape id='particle_{id + 1}' type="ply">
                   <string name="filename" value="{mesh}"/>
                   <ref id="particle_{id + 1}_bsdf" name="bsdf"/>
                   <transform name="to_world">
                       <translate value="0. 0. 0."/>
                       <scale value="{mesh_scale}"/>
                   </transform>
               </shape>
           """

    return (
        len(mesh_dir), d_particle_keys, d_stage_keys,
        f"""
        <scene version="3.0.0">
            <integrator type='direct_projective'/>

            {create_sensor(sample_params_dict['gaussdev'])}

            <shape id='emission1' type="ply">           
                <string name="filename" value="{MESH_PATH}/light_source.ply"/>
                <emitter type="area">
                    <rgb name="radiance" value="{emission1}"/>
                </emitter>
                <transform name="to_world">
                   <rotate x="1" angle="-90"/>
                    <translate value="0.00 0.00 0.00"/>
                    <scale value ="15 200 15"/>
                </transform>
            </shape>

            <shape id='stage' type="ply">           
                <string name="filename" value="{MESH_PATH}/stage.ply"/>
                <bsdf type="principled" id="stage_bsdf">
                   <rgb name="base_color" value="{sample_params_dict['p_base_color']}"/>
                   <float name="roughness" value="{sample_params_dict['p_roughness']}" />
                   <float name="anisotropic" value="{sample_params_dict['p_anisotropic']}" />
                   <float name="metallic" value="{sample_params_dict['p_metallic']}" />
                   <float name="spec_trans" value="{sample_params_dict['p_spec_trans']}" />
                   <float name="specular" value="{sample_params_dict['p_specular']}" />
                   <float name="spec_tint"  value="{sample_params_dict['p_spec_tint']}" />
                   <float name="sheen" value="{sample_params_dict['p_sheen']}" />
                   <float name="sheen_tint" value="{sample_params_dict['p_sheen_tint']}" />
                   <float name="flatness" value="{sample_params_dict['p_flatness']}" />
                   <float name="clearcoat" value="{sample_params_dict['p_clearcoat']}" />
                   <float name="clearcoat_gloss" value="{sample_params_dict['p_clearcoat_gloss']}" />
               </bsdf>
                <transform name="to_world">

                    <translate value="0.00 -4.00 0.00"/>
                    <scale value ="10 1 10"/>
                </transform>
            </shape>

           <shape id='emission2' type="ply">           
                <string name="filename" value="{MESH_PATH}/light_source2.ply"/>
                <emitter type="area">
                    <rgb name="radiance" value="{emission2}"/>
                </emitter>
                <transform name="to_world">
                    <rotate x="1" angle="-180"/>
                    <translate value="0.00 8.00 0.00"/>
                    <scale value ="10 1 10"/>
                </transform>
            </shape>

            <!-- Shapes -->
            {particle_meshes}

        </scene>
    """
    )
