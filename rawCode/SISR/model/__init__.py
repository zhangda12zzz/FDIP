def create_model(args, character_names, dataset, std_paths=None):
    if args.model == 'mul_top_mul_ske':
        args.skeleton_info = 'concat'
        import model.architecture
        return model.architecture.GAN_model(args, character_names, dataset, std_paths)

    else:
        raise Exception('Unimplemented model')
    
def create_CIPmodel(args, dataset, std_paths=None, log_path=None):
    if args.model == 'mul_top_mul_ske':
        args.skeleton_info = 'concat'
        import model.architecture_CIP
        return model.architecture_CIP.GAN_model_CIP(args, dataset, std_paths, log_path=log_path)

    else:
        raise Exception('Unimplemented model')
