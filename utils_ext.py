def save_model(net, opt):
    save_path = os.path.join(opt.save_path, opt.model_name)
    torch.save(net.state_dict(), save_path)
    structure_path = os.path.join(opt.save_path, opt.model_name+".txt")
    with open(structure_path, 'w') as sys.stdout:
        print("Note: ", opt.note)
        print("==============")
        with open(opt.feature_name_path, 'r') as f:
            for line in f:
                print(line)
        print("==============")
        net.print()


def load_model(model, opt):
    save_path = os.path.join(opt.load_path, opt.model_name)
    model.load_state_dict(torch.load(save_path))
    return model